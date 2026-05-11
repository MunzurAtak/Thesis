from src.pipeline.prompting_pipeline import run_full_prompting_debug_pipeline


def main():
    run_full_prompting_debug_pipeline()


if __name__ == "__main__":
    import time
    _t0 = time.time()
    main()
    print(f"\nCompleted in {time.time() - _t0:.1f}s")
