"""In-process Jupyter Notebook (.ipynb) executor without TCP/ZeroMQ socket binding."""

from __future__ import annotations

import ast
import base64
import builtins
import io
import sys
import time
from pathlib import Path
from typing import Any

from . import config as _cfg  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf


class _TeeStdout(io.TextIOBase):
    """Write live output to terminal while recording clean lines for notebook cell output."""

    def __init__(self, real_stream: Any) -> None:
        self._real = real_stream
        self._chunks: list[str] = []

    def write(self, s: str) -> int:
        self._real.write(s)
        self._real.flush()
        self._chunks.append(s)
        return len(s)

    def flush(self) -> None:
        self._real.flush()

    def get_clean_text(self) -> str:
        raw = "".join(self._chunks)
        cleaned_lines: list[str] = []
        for line in raw.split("\n"):
            if "\r" in line:
                parts = [p for p in line.split("\r") if p.strip()]
                line = parts[-1] if parts else ""
            if "\x08" in line:
                buf: list[str] = []
                for ch in line:
                    if ch == "\x08":
                        if buf:
                            buf.pop()
                    else:
                        buf.append(ch)
                line = "".join(buf)
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)


def execute_notebook_inplace(nb_path: Path) -> None:
    """Execute all code cells of `nb_path` in-process and write validated outputs back."""
    print(f"\n{'=' * 72}\n[nb_runner] Executing {nb_path.name}\n{'=' * 72}")
    t_start = time.perf_counter()

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    cell_outputs_collector: list[nbf.NotebookNode] = []

    def custom_display(*objs: Any) -> None:
        for obj in objs:
            data: dict[str, Any] = {"text/plain": repr(obj)}
            if hasattr(obj, "_repr_html_"):
                html = obj._repr_html_()
                if html:
                    data["text/html"] = html
            elif hasattr(obj, "to_html"):
                data["text/html"] = obj.to_html()
            cell_outputs_collector.append(
                nbf.v4.new_output(output_type="display_data", data=data, metadata={})
            )

    def custom_plt_show(*args: Any, **kwargs: Any) -> None:
        for num in plt.get_fignums():
            fig = plt.figure(num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
            b64_png = base64.b64encode(buf.getvalue()).decode("ascii")
            w_px, h_px = (int(fig.get_figwidth() * 140), int(fig.get_figheight() * 140))
            cell_outputs_collector.append(
                nbf.v4.new_output(
                    output_type="display_data",
                    data={
                        "image/png": b64_png,
                        "text/plain": f"<Figure {w_px}x{h_px}>",
                    },
                    metadata={},
                )
            )
            plt.close(fig)

    orig_plt_show = plt.show
    plt.show = custom_plt_show
    setattr(builtins, "display", custom_display)

    ns: dict[str, Any] = {"__name__": "__main__", "display": custom_display}
    exec_count = 0

    try:
        for idx, cell in enumerate(nb.cells, start=1):
            if cell.cell_type != "code" or not cell.source.strip():
                continue

            exec_count += 1
            cell_outputs_collector = []
            print(f"\n--- [{nb_path.name}] Cell #{idx} (exec {exec_count}) ---")

            tee_out = _TeeStdout(sys.__stdout__)
            tee_err = _TeeStdout(sys.__stderr__)
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = tee_out, tee_err

            try:
                tree = ast.parse(cell.source, filename=f"{nb_path.name}_cell_{exec_count}")
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last_expr = ast.Expression(tree.body.pop().value)
                    if tree.body:
                        exec(
                            compile(tree, f"{nb_path.name}_cell_{exec_count}", "exec"),
                            ns,
                        )
                    val = eval(
                        compile(last_expr, f"{nb_path.name}_cell_{exec_count}", "eval"),
                        ns,
                    )
                    if val is not None:
                        data: dict[str, Any] = {"text/plain": repr(val)}
                        if hasattr(val, "_repr_html_"):
                            html = val._repr_html_()
                            if html:
                                data["text/html"] = html
                        cell_outputs_collector.append(
                            nbf.v4.new_output(
                                output_type="execute_result",
                                data=data,
                                execution_count=exec_count,
                            )
                        )
                else:
                    exec(compile(tree, f"{nb_path.name}_cell_{exec_count}", "exec"), ns)
            finally:
                sys.stdout, sys.stderr = old_out, old_err

            stdout_txt = tee_out.get_clean_text()
            stderr_txt = tee_err.get_clean_text()

            final_outputs: list[nbf.NotebookNode] = []
            if stdout_txt.strip():
                final_outputs.append(
                    nbf.v4.new_output(output_type="stream", name="stdout", text=stdout_txt)
                )
            if stderr_txt.strip():
                final_outputs.append(
                    nbf.v4.new_output(output_type="stream", name="stderr", text=stderr_txt)
                )
            final_outputs.extend(cell_outputs_collector)

            cell.execution_count = exec_count
            cell.outputs = final_outputs

        nbf.validate(nb)
        with open(nb_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)

        elapsed = time.perf_counter() - t_start
        print(f"\n[nb_runner] Completed and validated {nb_path.name} in {elapsed:.1f}s.")
    finally:
        plt.show = orig_plt_show
        plt.close("all")
        ns.clear()
        from .config import cleanup_temp_prefix_cache
        from .trainer import release_memory

        cleanup_temp_prefix_cache()
        release_memory()


def main() -> None:
    targets = [Path(arg).resolve() for arg in sys.argv[1:]]
    if not targets:
        raise SystemExit("Usage: python -m src.nb_runner <notebook1.ipynb> [...]")
    for nb_file in targets:
        execute_notebook_inplace(nb_file)


if __name__ == "__main__":
    main()

