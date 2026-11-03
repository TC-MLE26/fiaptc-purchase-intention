"""Console helpers shared by the pipeline entrypoints."""

import sys


def configure_utf8_console() -> None:
    """Reconfigure stdout and stderr to UTF-8, replacing unencodable characters.

    Windows consoles often use cp1252, which cannot encode the emoji that
    MLflow prints (for example in "View run"), breaking the stage with a
    ``UnicodeEncodeError``. Streams that cannot be reconfigured, such as the
    ones captured by pytest, are left untouched.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
