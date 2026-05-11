from src.pipeline.prompting_pipeline import run_full_prompting_debug_pipeline


def main():
    run_full_prompting_debug_pipeline(
        config_path="configs/prompting_ollama_judge_smoke_test.json",
        transcript_dir="outputs/transcripts",
        judge_score_dir="outputs/judge_scores",
        metrics_dir="outputs/metrics",
        metrics_output_path="outputs/metrics/prompting_ollama_judge_smoke_test_metrics.csv",
    )


if __name__ == "__main__":
    import time
    _t0 = time.time()
    main()
    print(f"\nCompleted in {time.time() - _t0:.1f}s")
