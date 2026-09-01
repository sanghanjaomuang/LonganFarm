import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="วิเคราะห์ดิน+พยากรณ์ผลผลิตลำไย")
st.title("ระบบวิเคราะห์สภาพดินและพยากรณ์ผลผลิตลำไย")

# ---------- ข้อมูลตัวอย่างสวนลำไย 80 สวน (แทนด้วยข้อมูลจริงได้) ----------
df = pd.read_csv("lab07-longan-soil.csv")
X = df.drop(columns=["yield"])
y = df["yield"]


@st.cache_resource
def สอนโมเดล():
    return RandomForestRegressor(n_estimators=200, random_state=42).fit(X, y)


model = สอนโมเดล()

# ===== ส่วนที่ 1: วิเคราะห์สภาพดิน =====
st.header("1) วิเคราะห์สภาพดิน")
c1, c2 = st.columns(2)
pH = c1.number_input("ค่า pH ดิน", 3.0, 9.0, 6.0, 0.1)
N  = c1.number_input("ไนโตรเจน N (mg/kg)", 0, 200, 40)
P  = c1.number_input("ฟอสฟอรัส P (mg/kg)", 0, 200, 25)
K  = c2.number_input("โพแทสเซียม K (mg/kg)", 0, 400, 180)
OM = c2.number_input("อินทรียวัตถุ OM (%)", 0.0, 10.0, 2.5, 0.1)
moisture = c2.number_input("ความชื้นดิน (%)", 0, 100, 55)

st.subheader("คำแนะนำจัดการดิน (สำหรับลำไย)")
if pH < 5.5:
    st.warning("ดินเป็นกรดเกินไป → ใส่ปูนขาวปรับ pH (ลำไยชอบ 5.5–6.5)")
elif pH > 6.5:
    st.warning("ดินค่อนข้างด่าง → ระวังธาตุอาหารบางตัวถูกตรึง")
else:
    st.success("pH เหมาะกับลำไย")
if OM < 2.0:
    st.write("• อินทรียวัตถุต่ำ → เพิ่มปุ๋ยคอก/ปุ๋ยหมัก")
if N < 30:
    st.write("• ไนโตรเจนต่ำ → บำรุงใบด้วยปุ๋ย N")
if K < 150:
    st.write("• โพแทสเซียมต่ำ → สำคัญต่อการติดผลลำไย ควรเพิ่ม K")

# ===== ส่วนที่ 2: พยากรณ์ผลผลิต =====
st.header("2) พยากรณ์ผลผลิตลำไย")
c3, c4 = st.columns(2)
temp = c3.number_input("อุณหภูมิเฉลี่ย (°C)", 15, 40, 27)
rain = c3.number_input("ปริมาณน้ำฝน (มม./ปี)", 0, 3000, 1200)
density  = c4.number_input("ความหนาแน่น (ต้น/ไร่)", 10, 100, 45)
chlorate = c4.number_input("โพแทสเซียมคลอเรต (กก./ไร่)", 0.0, 30.0, 8.0, 0.1)

x_new = pd.DataFrame([[pH, N, P, K, OM, moisture, temp, rain, density, chlorate]],
                     columns=X.columns)
yhat = model.predict(x_new)[0]
st.metric("ผลผลิตคาดการณ์", f"{yhat:.0f} กก./ไร่")

# ปัจจัยสำคัญ — ดูจากค่าสหสัมพันธ์ (correlation)
st.subheader("ปัจจัยที่สัมพันธ์กับผลผลิตมากที่สุด")
corr = df.corr()
st.bar_chart(corr["yield"].drop("yield").sort_values(ascending=False))
st.caption("ค่าใกล้ +1 = ปัจจัยเพิ่ม ผลผลิตเพิ่มตาม · ใกล้ -1 = ปัจจัยเพิ่ม ผลผลิตลด · ใกล้ 0 = ไม่เกี่ยวกัน")

with st.expander("ดูตารางสหสัมพันธ์ทั้งหมด (correlation matrix)"):
    st.dataframe(corr.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1)
                           .format("{:.2f}"))
    ระหว่างปัจจัย = corr.drop(columns=["yield"]).drop(index=["yield"]).abs().values
    ระหว่างปัจจัย = ระหว่างปัจจัย[ระหว่างปัจจัย < 0.999]
    st.caption(
        f"ปัจจัยด้วยกันเองสัมพันธ์กันเฉลี่ยเพียง {ระหว่างปัจจัย.mean():.2f} "
        "แปลว่าแต่ละปัจจัยแปรผันอิสระต่อกัน จึงแยกได้ว่าตัวไหนมีผลต่อผลผลิตจริง"
    )

# ===== ส่วนที่ 3: โมเดลนี้แม่นแค่ไหน =====
st.header("3) ตรวจสอบความแม่นของโมเดล")
st.write(
    "แบ่งข้อมูลเป็น 5 กลุ่ม ปิดทีละกลุ่มแล้วให้โมเดลที่ไม่เคยเห็นกลุ่มนั้นทายดู "
    "(5-fold cross-validation) จะได้รู้ว่าเวลาเจอสวนใหม่จริง ๆ โมเดลพลาดประมาณเท่าไหร่"
)


@st.cache_data
def ตรวจสอบย้อนหลัง():
    m = RandomForestRegressor(n_estimators=200, random_state=42)
    return cross_val_predict(m, X, y, cv=KFold(5, shuffle=True, random_state=42))


y_check = ตรวจสอบย้อนหลัง()

c5, c6 = st.columns(2)
c5.metric("พลาดเฉลี่ย (MAE)", f"{mean_absolute_error(y, y_check):.0f} กก./ไร่")
c6.metric("R² (คะแนนความแม่น)", f"{r2_score(y, y_check):.3f}")

ตาราง = pd.DataFrame({
    "ผลผลิตจริง": y,
    "โมเดลทาย": y_check.round(0),
    "ห่างจากจริง": (y_check - y).round(0),
})
st.write("**เทียบผลผลิตจริง กับ ที่โมเดลทาย ทั้ง 80 สวน**")
st.scatter_chart(ตาราง, x="ผลผลิตจริง", y="โมเดลทาย")
st.caption("จุดยิ่งเรียงเป็นเส้นทแยงมุม แปลว่าโมเดลยิ่งแม่น")
st.dataframe(ตาราง, width="stretch", height=280)
