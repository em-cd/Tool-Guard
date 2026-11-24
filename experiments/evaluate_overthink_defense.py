#!/usr/bin/env python3
"""
Evaluate overthinking-defense variants on top of Tool-Guard.

Variants: alignment_prompt | judge
Conditions per variant (names reused from the existing script so
ot.summarize / ot.aggregate_results work unchanged):
    B_benign_TG     benign tasks + variant        (utility / false positives)
    D_overthink_TG  overthinking tools + variant  (effectiveness)

Rows A-D from evaluate_split_replan_overthink.py are the baselines; do not rerun
them, but use the same model, tasks, repeats and MAX_ITERS.

Usage:
  python experiments/evaluate_overthink_variants.py --provider <p> --model <m> --suite banking \
      --all-tasks --repeats 3 --variants alignment_prompt judge --output results/variants.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_split_replan_defense as driver
import evaluate_split_replan_overthink as ot
from agentdojo.agent_pipeline.overthink_defense import OverthinkDefense, ProgressMonitoringExecutor

from agentdojo.agent_pipeline import AgentPipeline, InitQuery, SystemMessage, ToolsExecutor
from agentdojo.agent_pipeline.split_replan_defense import SplitReplanToolsExecutionLoop
from agentdojo.agent_pipeline.token_tracking import TokenTracker
from agentdojo.task_suite import get_suite

CONDS = [("B_benign_TG", False), ("D_overthink_TG", True)]

def build_variant(model, suite_name, tracker, inject, variant, provider, k, verbose):
    client, llm_class, _ = driver.get_provider_client_and_llm(provider, model)
    head = [SystemMessage(driver.SYSTEM_PROMPT.format(suite_name=suite_name)), InitQuery()]
    if inject is not None:
        head.append(inject)
    llm = driver._create_llm_instance(llm_class, client, model, tracker)
    defense = OverthinkDefense(
        client=driver._validator_client(client, llm_class, tracker),
        model=model,
        mode=variant,
        k=k,
        verbose=verbose,
    )
    loop = SplitReplanToolsExecutionLoop(
        llm=llm,
        tools_executor=ProgressMonitoringExecutor(ToolsExecutor(), defense),
        defense=defense,
        max_iters=ot.MAX_ITERS,
        include_terminate="term" in variant.split("+")
    )
    return AgentPipeline(head + [loop]), defense


def key_of(r):
    return (r["task"], r["rep"], r["variant"], r["cond"])


def run(args):
    suite = get_suite("v1.2.2", args.suite)
    all_ids = list(suite.user_tasks.keys())
    task_ids = args.tasks or (all_ids if args.all_tasks else all_ids[: args.num_tasks])

    task_ids = args.tasks or (all_ids if args.all_tasks else all_ids[: args.num_tasks])

    n_runs = len(task_ids) * args.repeats * len(args.variants) * len(CONDS)
    print("=" * 70)
    print(f"Suite:    {args.suite}  ({args.provider} / {args.model})")
    print(f"Variants: {', '.join(args.variants)}")
    print(f"Tasks:    {len(task_ids)}  ({', '.join(task_ids)})")
    print(f"Repeats:  {args.repeats}")
    print(f"Conds:    {', '.join(c for c, _ in CONDS)}")
    print(f"Total:    {n_runs} runs (k={args.k})")
    print("=" * 70)

    tracker = TokenTracker()
    ckpt = args.output + ".checkpoint.json"
    records = ot.load_checkpoint(ckpt)
    done = {key_of(r) for r in records if r.get("error") is None}

    for task_id in task_ids:
        user_task = suite.get_user_task_by_id(task_id)
        for rep in range(args.repeats):
            for variant in args.variants:
                for cond, attack in CONDS:
                    key = (task_id, rep, variant, cond)
                    if key in done:
                        print(f"[skip] {key}")
                        continue
                    print(f"\n[{task_id}] rep={rep} variant={variant} cond={cond}")

                    injection = ot.OverthinkInjection(ot.CycleState()) if attack else None
                    pipeline, defense = build_variant(
                        args.model, args.suite, tracker, injection, variant,
                        args.provider, args.k, not args.quiet,
                    )
                    utility, messages, stats, error, latency, tokens = driver.run_single_task(
                        pipeline, suite, user_task, defense, tracker=tracker, label=f"{variant}/{cond}",
                        debug=args.debug,
                    )
                    record = {
                        "task": task_id,
                        "prompt": user_task.PROMPT,
                        "rep": rep,
                        "variant": variant,
                        "cond": cond,
                        "k": args.k,
                        "utility": utility,
                        "error": error,
                        "latency_s": latency,
                        "tokens": tokens,
                        "validation_log": defense.stats.validation_log,
                        "stalled_tools": sorted(defense.stalled),
                        "judge_calls": defense.judge_calls,
                        "progress_log": defense.progress_log,
                        "terminate_votes": defense.stats.terminate_votes,
                        "unjudged_terminations": defense.stats.unjudged_terminations,
                        "termination_checks": defense.termination_checks,
                        "termination_log": defense.termination_log,
                        "removed_tools": sorted(defense.removed_tools),
                        **ot.summarize(messages, stats, cond),
                    }
                    records = [r for r in records if key_of(r) != key]
                    records.append(record)
                    ot.save_checkpoint(ckpt, records)
                    if error is None:
                        done.add(key)
                    print(f"      Utility: {utility}, Tokens: {tokens['total']:,}, "
                          f"Tools: {record['executed_sequence']}")

    summary = {v: ot.aggregate_results([r for r in records if r["variant"] == v]) for v in args.variants}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {"config": vars(args), "tasks": task_ids, "summary": summary, "results": records},
            f, indent=2, default=str,
        )
    print(f"\nSaved to {args.output}")
    for v, s in summary.items():
        for cond, d in s.items():
            print(f"{v:10} {cond:16} utility {d['utility']['rate']:.1%} | "
                  f"tokens {d['tokens']['total']['mean']:,.0f} | tool calls {d['tool_calls']['mean']:.1f} | "
                  f"cap hits {d['iteration_cap_hits']}/{d['n']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--provider", default="openai")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--suite", default="banking")
    p.add_argument("--num-tasks", type=int, default=5)
    p.add_argument("--tasks", nargs="+")
    p.add_argument("--all-tasks", action="store_true")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--variants", nargs="+",
               default=["narrow", "judge", "term", "remove", "narrow+term", "narrow+remove", "narrow+term+remove"],
               choices=["narrow", "judge", "term", "remove", "narrow+term", "narrow+remove",
                        "narrow+term+remove", "term+nojudge", "narrow+term+nojudge"])
    p.add_argument("--k", type=int, default=2, help="no-progress calls before a tool is removed")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--output", required=True)
    run(p.parse_args())