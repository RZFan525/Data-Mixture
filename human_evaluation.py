import json
import streamlit as st
from pathlib import Path
import textwrap

st.set_page_config(page_title="Error Case Viewer", layout="wide")
st.title("Human Evaluation: Error Case Viewer")

# ---------- sidebar: load ----------
st.sidebar.header("Load JSON")
json_path_str = st.sidebar.text_input("Path to error cases JSON", value="error_cases.json")

path = Path(json_path_str)
if not path.exists():
    st.warning(f"File not found: {path}")
    st.stop()

try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception as e:
    st.error(f"Failed to read JSON: {e}")
    st.stop()

if not isinstance(data, list) or len(data) == 0:
    st.error("JSON must be a non-empty list of objects.")
    st.stop()

n = len(data)
max_show = n

# ---------- sidebar: sample picker ----------
st.sidebar.header("Sample Picker")

# 下拉：点一下选，支持键盘输入过滤
labels = []
for i in range(max_show):
    item = data[i]
    ds = str(item.get("dataset", ""))
    _id = str(item.get("id", ""))
    labels.append(f"{i+1:03d} | {ds} | {_id}")

picked_label = st.sidebar.selectbox("Pick sample", options=labels, index=0)

# 手动输入：精确定位
picked_num = st.sidebar.number_input(
    "Or type sample index",
    min_value=1,
    max_value=max_show,
    value=int(picked_label.split("|")[0]),
    step=1,
)

idx = int(picked_num) - 1
item = data[idx]

# ---------- helpers ----------
def get_str(x, default=""):
    return default if x is None else str(x)

def render_multiline_text(text: str):
    """Render as wrapped text (no horizontal scrolling)."""
    text = get_str(text)
    if not text:
        st.write("")
        return
    # 用 markdown + <pre> 强制保留换行，同时支持自动换行
    st.markdown(
        f"""
        <div style="white-space: pre-wrap; word-break: break-word; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 0.95rem; padding: 0.75rem; border: 1px solid rgba(49, 51, 63, 0.2); border-radius: 0.5rem;">
        {text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- display ----------
st.subheader(f"Sample {idx+1} / {n}  (showing 1-{max_show})")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Metadata")
    st.write("**dataset:**", get_str(item.get("dataset", "")))
    st.write("**id:**", get_str(item.get("id", "")))
    st.write("**answer (gold):**", get_str(item.get("answer", "")))
    st.write("**prediction:**", get_str(item.get("prediction", "")))
    st.write("**accuracy:**", item.get("accuracy", None))

    st.markdown("### Options")
    options = item.get("options", [])
    if isinstance(options, list) and options:
        letters = ["A", "B", "C", "D"]
        for i, opt in enumerate(options):
            letter = letters[i] if i < len(letters) else str(i)
            st.write(f"**{letter}.** {get_str(opt)}")
    else:
        st.write("(no options)")

with col2:
    st.markdown("### Model Output (wrapped)")
    render_multiline_text(item.get("model_output", ""))

    st.markdown("### Messages (optional)")
    with st.expander("Show messages"):
        msgs = item.get("messages", [])
        if isinstance(msgs, list):
            for m in msgs:
                role = m.get("role", "unknown")
                content = get_str(m.get("content", ""))
                st.markdown(f"**{role}**")
                render_multiline_text(content)
        else:
            st.write("(messages not a list)")

    with st.expander("Show prompt"):
        render_multiline_text(item.get("prompt", ""))

st.divider()
