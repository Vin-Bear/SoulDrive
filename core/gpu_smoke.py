import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        return 2

    model_path = args[0]
    try:
        n_gpu_layers = int(args[1])
        n_ctx = int(args[2])
    except ValueError:
        return 2

    try:
        from llama_cpp import Llama

        llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=max(1024, min(n_ctx, 4096)),
            verbose=False,
        )
        llm.create_chat_completion(
            messages=[{"role": "user", "content": "用一句话回答：Transformer是什么？"}],
            max_tokens=8,
            temperature=0.0,
        )
        close_fn = getattr(llm, "close", None)
        if callable(close_fn):
            close_fn()
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
