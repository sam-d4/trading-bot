import os

# stable-baselines3 imports matplotlib (for optional plot logging we don't use) at import time,
# which on some machines crashes or hangs scanning system fonts (observed here: `system_profiler
# SPFontsDataType -json` returns an empty result, and matplotlib's macOS font-list parser doesn't
# handle that gracefully - KeyError: '_items'). We never render matplotlib plots in this project,
# so it's safe to just skip system font discovery entirely. Must be set before stable_baselines3
# (or matplotlib) is imported anywhere, hence living at the top of this package's __init__.
os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")
