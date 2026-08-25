"""
trace.py
========
Helper untuk mencetak JEJAK EKSEKUSI (trace) langkah demi langkah ke terminal.

Format yang dicetak:
    Running <file>.py function <fungsi>
    Input example: <contoh input yang diterima>
    Running..
    Output example: <contoh output yang dihasilkan>

Modul ini ringan (hanya memakai print) dan tidak bergantung pada modul lain,
sehingga bisa dipanggil dari get_example/, llm_inference/, maupun post_processing/.
"""


def _truncate(value, limit: int = 90) -> str:
    """Potong teks panjang agar jejak tetap rapi di terminal."""
    s = str(value)
    if len(s) <= limit:
        return s
    return s[:limit] + "…"


def trace_start(file_name: str, func_name: str) -> None:
    """Cetak penanda bahwa sebuah fungsi mulai dieksekusi."""
    print(f"\nRunning {file_name}.py function {func_name}")


def trace_input(example) -> None:
    """Cetak contoh input yang diterima fungsi."""
    print(f"Input example: {_truncate(example)}")


def trace_running() -> None:
    """Cetak penanda proses sedang berjalan."""
    print("Running..")


def trace_output(example) -> None:
    """Cetak contoh output yang dihasilkan fungsi."""
    print(f"Output example: {_truncate(example)}")
