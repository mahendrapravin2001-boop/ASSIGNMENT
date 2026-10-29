"""
JIE42403 Scientific Visualisation - Assignment
Household Income, Expenditure and Population in Malaysia (2022 vs 2024)

Three data sets:
  1. hh_income_parlimen.csv              - household income by parliamentary seat (RM / month)
  2. hh_expenditure_parlimen.csv         - household expenditure by parliamentary seat (RM / month)
  3. population_state_2019_2022_2024.csv - population by state, sex, age, ethnicity ('000 persons)

Timeline used: 2022 and 2024 only.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).parent
YEARS = [2022, 2024]
Y0, Y1 = YEARS
YEAR_COLORS = {"2022": "#4C78A8", "2024": "#F58518"}   # blue = 2022, orange = 2024
C_INCOME, C_EXP = "#2A9D8F", "#8E5BA8"                 # teal = income, purple = expenditure
TEMPLATE = "plotly_white"

AGE_ORDER = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44",
             "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85+"]
WORKING_AGES = AGE_ORDER[3:13]   # 15-19 ... 60-64

# Approximate centre point of each state (latitude, longitude) for the map
STATE_XY = {
    "Johor": (1.95, 103.35), "Kedah": (6.10, 100.55), "Kelantan": (5.40, 102.05),
    "Melaka": (2.20, 102.25), "Negeri Sembilan": (2.75, 102.20), "Pahang": (3.85, 102.55),
    "Perak": (4.70, 101.10), "Perlis": (6.45, 100.20), "Pulau Pinang": (5.40, 100.38),
    "Sabah": (5.45, 117.00), "Sarawak": (2.60, 113.50), "Selangor": (3.35, 101.45),
    "Terengganu": (5.00, 103.00), "W.P. Kuala Lumpur": (3.14, 101.69),
    "W.P. Labuan": (5.28, 115.24), "W.P. Putrajaya": (2.93, 101.69),
}

st.set_page_config(page_title="Malaysia Income, Spending & Population",
                   page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; max-width: 1400px;}
      .objective {background:#eef4fb; border-left:6px solid #4C78A8; padding:.8rem 1.1rem;
                  border-radius:6px; font-size:1.05rem; margin-bottom:.8rem;}
      .interp {background:#fafafa; border:1px solid #e3e3e3; border-radius:8px;
               padding:1rem 1.3rem; margin-top:.6rem;}
      .small {color:#666; font-size:.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- data
@st.cache_data
def load_data():
    inc = pd.read_csv(HERE / "hh_income_parlimen.csv")
    exp = pd.read_csv(HERE / "hh_expenditure_parlimen.csv")
    pop = pd.read_csv(HERE / "population_state_2019_2022_2024.csv")
    for d in (inc, exp, pop):
        d["year"] = pd.to_datetime(d["date"]).dt.year
    seats = inc.merge(exp[["date", "parlimen", "expenditure"]], on=["date", "parlimen"])
    seats = seats[seats["year"].isin(YEARS)].copy()
    seats["year_s"] = seats["year"].astype(str)
    seats["spend_ratio"] = seats["expenditure"] / seats["income_mean"] * 100
    seats["income_gap"] = seats["income_mean"] / seats["income_median"]
    pop = pop[pop["year"].isin(YEARS)].copy()
    return seats, pop


def state_table(seats):
    """Average of the parliamentary seats inside each state, per year."""
    return seats.groupby(["state", "year"], as_index=False).agg(
        income_median=("income_median", "mean"), income_mean=("income_mean", "mean"),
        expenditure=("expenditure", "mean"), spend_ratio=("spend_ratio", "mean"),
        income_gap=("income_gap", "mean"), n_seats=("parlimen", "nunique"))


def growth(df, col):
    """% change 2022 -> 2024 by state for a state_table column."""
    p = df.pivot(index="state", columns="year", values=col)
    return ((p[Y1] / p[Y0] - 1) * 100).rename(col)


def pct(a, b):
    return (b / a - 1) * 100


def population_overall(pop):
    return pop[(pop.sex == "both") & (pop.age == "overall") & (pop.ethnicity == "overall")]


seats_all, pop_all = load_data()
all_states = sorted(seats_all["state"].unique())

# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to objective",
    ["1 · Income & spending growth", "2 · Income vs spending", "3 · Population & geography"],
)
st.sidebar.markdown("---")
st.sidebar.subheader("Filter")
chosen = st.sidebar.multiselect("States / territories", all_states, default=all_states)
if not chosen:
    st.warning("Please select at least one state in the sidebar.")
    st.stop()

seats = seats_all[seats_all["state"].isin(chosen)]
pop = pop_all[pop_all["state"].isin(chosen)]
st_tab = state_table(seats)
n_seats = seats["parlimen"].nunique()

with st.sidebar.expander("ℹ️ About the data"):
    st.markdown(
        """
        **Data sets (Malaysia)**
        - Household income, 222 parliamentary seats
        - Household expenditure, 222 parliamentary seats
        - Population by state, sex, age, ethnicity

        **Timeline:** 2022 and 2024 only.

        **Notes**
        - Income & expenditure are in RM per month.
        - State values = average of the seats in that state.
        - *Spending ratio* = expenditure ÷ mean income × 100.
        - Population is in thousands of persons.
        """
    )

st.title("Household Income, Spending and Population in Malaysia")
st.caption(f"Comparing 2022 and 2024  ·  {len(chosen)} state(s)  ·  {n_seats} parliamentary seats selected")


SHORT = {"W.P. Kuala Lumpur": "Kuala Lumpur", "W.P. Putrajaya": "Putrajaya", "W.P. Labuan": "Labuan",
         "Negeri Sembilan": "N. Sembilan", "Pulau Pinang": "Penang"}


def short(name):
    return SHORT.get(name, name)


def summary_row(items):
    """Each item: (label, value, delta, help[, colour]); colour = normal | inverse | off."""
    cols = st.columns(len(items))
    for c, item in zip(cols, items):
        label, value, delta, help_ = item[:4]
        colour = item[4] if len(item) > 4 else ("normal" if delta and delta[0] in "+-" else "off")
        c.metric(label, value, delta, delta_color=colour, delta_arrow="off" if colour == "off" else "auto",
                 border=True, help=help_)


def objective(text):
    st.markdown(f'<div class="objective"><b>Objective:</b> {text}</div>', unsafe_allow_html=True)


def interpretation(points, meaning):
    html = "<ul>" + "".join(f"<li>{p}</li>" for p in points) + "</ul>"
    st.markdown(f'<div class="interp"><h4>Interpretation / Discussion</h4>{html}'
                f"<p><b>Scientific meaning:</b> {meaning}</p></div>", unsafe_allow_html=True)


def style(fig, h=400):
    fig.update_layout(template=TEMPLATE, height=h, margin=dict(l=10, r=10, t=55, b=10),
                      title_font_size=15, legend_title_text="")
    return fig


# ============================================================================= PAGE 1
if page.startswith("1"):
    st.header("Objective 1 · Income & spending growth, 2022 → 2024")
    objective("To measure how household income and household expenditure changed between 2022 and "
              "2024, and to identify which states grew fastest.")

    a, b = seats[seats.year == Y0], seats[seats.year == Y1]
    g_med = growth(st_tab, "income_median")
    g_exp = growth(st_tab, "expenditure")
    summary_row([
        (f"Avg median income {Y1}", f"RM {b.income_median.mean():,.0f}",
         f"{pct(a.income_median.mean(), b.income_median.mean()):+.1f}% vs {Y0}", "Average of seat-level median incomes"),
        (f"Avg expenditure {Y1}", f"RM {b.expenditure.mean():,.0f}",
         f"{pct(a.expenditure.mean(), b.expenditure.mean()):+.1f}% vs {Y0}", "Average of seat-level household expenditure", "inverse"),
        ("Fastest income growth", short(g_med.idxmax()), f"{g_med.max():+.1f}%", "State with the highest % rise in median income"),
        ("Slowest income growth", short(g_med.idxmin()), f"{g_med.min():+.1f}%", "State with the lowest % change in median income"),
    ])

    c1, c2 = st.columns([3, 2])
    with c1:
        metrics = {"Median household income": "income_median", "Mean household income": "income_mean",
                   "Household expenditure": "expenditure"}
        label = st.selectbox("Indicator for the line plot", list(metrics))
        col = metrics[label]
        t = st_tab.assign(year_s=st_tab["year"].astype(str))
        fig = px.line(t, x="year_s", y=col, color="state", markers=True,
                      title=f"Chart 1 · {label} by state (RM / month)",
                      labels={"year_s": "Year", col: "RM / month", "state": "State"})
        avg = seats.groupby("year")[col].mean()
        fig.add_trace(go.Scatter(x=[str(y) for y in avg.index], y=avg.values, mode="lines+markers",
                                 name="Selected average", line=dict(color="black", width=4, dash="dash")))
        st.plotly_chart(style(fig, 450), width='stretch')
    with c2:
        gdf = pd.concat([g_med, g_exp], axis=1).reset_index().sort_values("income_median")
        gdf = gdf.melt(id_vars="state", value_vars=["income_median", "expenditure"],
                       var_name="Indicator", value_name="Growth")
        gdf["Indicator"] = gdf["Indicator"].map({"income_median": "Median income", "expenditure": "Expenditure"})
        fig = px.bar(gdf, y="state", x="Growth", color="Indicator", barmode="group", orientation="h",
                     color_discrete_map={"Median income": C_INCOME, "Expenditure": C_EXP},
                     title=f"Chart 2 · % growth {Y0}→{Y1} by state",
                     labels={"Growth": "Growth (%)", "state": ""})
        fig.add_vline(x=0, line_color="grey")
        st.plotly_chart(style(fig, 505), width='stretch')

    fig = px.histogram(seats, x="income_median", color="year_s", barmode="overlay", nbins=35,
                       opacity=0.65, marginal="box", color_discrete_map=YEAR_COLORS,
                       title="Chart 3 · Distribution of median household income across parliamentary seats",
                       labels={"income_median": "Median household income (RM / month)", "year_s": "Year"})
    st.plotly_chart(style(fig, 420), width='stretch')

    n_faster = int((g_exp > g_med).sum())
    interpretation([
        f"<b>Chart 1:</b> The average median income moved from RM {a.income_median.mean():,.0f} to "
        f"RM {b.income_median.mean():,.0f} "
        f"({pct(a.income_median.mean(), b.income_median.mean()):+.1f}%), while average expenditure moved from "
        f"RM {a.expenditure.mean():,.0f} to RM {b.expenditure.mean():,.0f} "
        f"({pct(a.expenditure.mean(), b.expenditure.mean()):+.1f}%). The gap between the top and bottom lines is large in both years, showing a persistent income gap between states.",
        f"<b>Chart 2:</b> {g_med.idxmax()} had the biggest rise in median income ({g_med.max():+.1f}%), "
        f"while {g_med.idxmin()} had the smallest ({g_med.min():+.1f}%). In {n_faster} of {len(g_med)} selected "
        f"states, expenditure grew faster than median income.",
        f"<b>Chart 3:</b> The 2024 histogram sits to the right of the 2022 one, so most seats earn more in 2024. "
        f"The long right tail and the box-plot outliers show a few very high-income seats, so the distribution is "
        f"right-skewed.",
    ], (f"Average spending grew {pct(a.expenditure.mean(), b.expenditure.mean()):+.1f}% and median income "
        f"{pct(a.income_median.mean(), b.income_median.mean()):+.1f}%. "
        + ("Spending grew clearly faster than income, so households have less left over each month. "
           if pct(a.expenditure.mean(), b.expenditure.mean()) > pct(a.income_median.mean(), b.income_median.mean()) + 1 else
           "Income and spending grew at a similar pace, so the monthly margin of a typical household stayed about the same. ")
        + "Nominal growth is not the same as real improvement, because prices also rose. Growth was uneven between "
          "states, which reflects differences in local wages, industries and cost of living."))


# ============================================================================= PAGE 2
elif page.startswith("2"):
    st.header("Objective 2 · Relationship between income and spending")
    objective("To examine how strongly household expenditure follows household income across parliamentary "
              "seats, and to find the places where spending takes the biggest share of income.")
    year = st.radio("Year for Charts 5 and 6", YEARS, index=1, horizontal=True)

    s_y = seats[seats.year == year]
    r_all = seats["income_median"].corr(seats["expenditure"])
    a, b = seats[seats.year == Y0], seats[seats.year == Y1]
    ratio_a, ratio_b = a.spend_ratio.mean(), b.spend_ratio.mean()
    top = s_y.loc[s_y.spend_ratio.idxmax()]
    chg = seats.pivot(index="parlimen", columns="year", values=["income_median", "expenditure"])
    faster = int(((chg["expenditure"][Y1] / chg["expenditure"][Y0]) > (chg["income_median"][Y1] / chg["income_median"][Y0])).sum())
    summary_row([
        ("Correlation (income vs spending)", f"r = {r_all:.2f}", None, "Pearson r between median income and expenditure, both years"),
        (f"Avg spending ratio {Y1}", f"{ratio_b:.1f}%", f"{ratio_b - ratio_a:+.1f} pts vs {Y0}", "Expenditure ÷ mean income × 100 (a rise means less left over)", "inverse"),
        (f"Highest ratio ({year})", f"{top.spend_ratio:.0f}%", top.parlimen, "Seat where spending takes the biggest share of income", "off"),
        ("Spending outgrew income", f"{faster} / {len(chg)}", None, "Number of seats whose expenditure growth beat median-income growth"),
    ])

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(seats, x="income_median", y="expenditure", color="year_s", hover_name="parlimen",
                         hover_data=["state"], color_discrete_map=YEAR_COLORS, opacity=0.75,
                         title=f"Chart 4 · Income vs expenditure per seat (r = {r_all:.2f})",
                         labels={"income_median": "Median income (RM / month)",
                                 "expenditure": "Expenditure (RM / month)", "year_s": "Year"})
        k, c0 = np.polyfit(seats["income_median"], seats["expenditure"], 1)
        xs = np.array([seats.income_median.min(), seats.income_median.max()])
        fig.add_trace(go.Scatter(x=xs, y=k * xs + c0, mode="lines", name=f"Trend (slope {k:.2f})",
                                 line=dict(color="black", dash="dash")))
        st.plotly_chart(style(fig, 470), width='stretch')
    with c2:
        st_y = st_tab[st_tab.year == year].set_index("state")
        cols = {"income_median": "Median income", "income_mean": "Mean income", "expenditure": "Expenditure",
                "spend_ratio": "Spending ratio", "income_gap": "Income gap"}
        z = st_y[list(cols)]
        zn = (z - z.min()) / (z.max() - z.min()).replace(0, 1)
        zn = zn.loc[z.sort_values("income_median", ascending=False).index]
        z = z.loc[zn.index]
        fmt = {"income_median": "{:,.0f}", "income_mean": "{:,.0f}", "expenditure": "{:,.0f}",
               "spend_ratio": "{:.1f}%", "income_gap": "{:.2f}×"}
        text = [[fmt[c].format(z.loc[s, c]) for c in cols] for s in z.index]
        fig = go.Figure(go.Heatmap(z=zn.values, x=list(cols.values()), y=list(zn.index), text=text,
                                   texttemplate="%{text}", colorscale="YlGnBu", showscale=True,
                                   colorbar=dict(title="Relative<br>level"),
                                   hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title=f"Chart 5 · State indicators heatmap, {year} (colour scaled per column)")
        st.plotly_chart(style(fig, 470), width='stretch')

    ranked = s_y.sort_values("spend_ratio")
    ext = pd.concat([ranked.head(10).assign(Group="Lowest 10"), ranked.tail(10).assign(Group="Highest 10")])
    fig = px.bar(ext, x="spend_ratio", y="parlimen", color="Group", orientation="h", hover_data=["state"],
                 color_discrete_map={"Lowest 10": "#2A9D8F", "Highest 10": "#E76F51"},
                 title=f"Chart 6 · Seats with the lowest and highest spending ratio, {year}",
                 labels={"spend_ratio": "Expenditure as % of mean income", "parlimen": ""})
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(style(fig, 560), width='stretch')

    hi10, lo10 = ranked.tail(10), ranked.head(10)
    hi_states = z["spend_ratio"].idxmax()
    lo_states = z["spend_ratio"].idxmin()
    interpretation([
        f"<b>Chart 4:</b> The points form a tight upward band (r = {r_all:.2f}) and the trend line has a slope of "
        f"{k:.2f}: each extra RM 1 of median income goes with about RM {k:.2f} of extra spending. The 2024 points "
        f"(orange) sit above and to the right of the 2022 points (blue).",
        f"<b>Chart 5:</b> In {year}, {hi_states} has the highest average spending ratio ({z.loc[hi_states, 'spend_ratio']:.1f}%) and "
        f"{lo_states} the lowest ({z.loc[lo_states, 'spend_ratio']:.1f}%). Dark rows at the top show high-income states; "
        f"the 'Income gap' column (mean ÷ median) shows where a few very high earners pull the mean above the median.",
        f"<b>Chart 6:</b> In {year}, spending takes up to {top.spend_ratio:.0f}% of income in {top.parlimen}. "
        f"The 10 highest-ratio seats have an average median income of RM {hi10.income_median.mean():,.0f}, compared "
        f"with RM {s_y.income_median.mean():,.0f} for all selected seats, so spending pressure is heaviest in "
        f"lower-income seats. The 10 lowest-ratio seats (RM {lo10.income_median.mean():,.0f}) include "
        f"{', '.join(sorted(lo10.state.unique()))}.",
    ], ("A strong positive correlation is expected: spending depends on what households earn. But the spending "
        "<i>ratio</i> is lower in higher-income seats, which matches Engel's law: poorer households spend a larger share "
        "of income on essentials, so they have less capacity to save. The average ratio "
        + ("rose clearly" if ratio_b - ratio_a > 1 else "rose slightly" if ratio_b - ratio_a > 0 else "fell")
        + f" between {Y0} and {Y1} ({ratio_a:.1f}% to {ratio_b:.1f}%)."))


# ============================================================================= PAGE 3
else:
    st.header("Objective 3 · Population and geography")
    objective("To show where Malaysia's population lives, how its age structure looks, and whether states with "
              "faster population growth also saw faster income growth.")
    year = st.radio("Year for Charts 7 and 8", YEARS, index=1, horizontal=True)

    p_all = population_overall(pop)
    pp = p_all.pivot(index="state", columns="year", values="population")
    p_g = pct(pp[Y0], pp[Y1])
    tot0, tot1 = pp[Y0].sum(), pp[Y1].sum()

    ages = pop[(pop.sex == "both") & (pop.ethnicity == "overall") & (pop.age.isin(AGE_ORDER))]
    ages_y = ages.groupby(["year", "age"])["population"].sum().unstack()
    work_share = {y: ages_y.loc[y, WORKING_AGES].sum() / ages_y.loc[y].sum() * 100 for y in YEARS}
    old_share = {y: ages_y.loc[y, AGE_ORDER[13:]].sum() / ages_y.loc[y].sum() * 100 for y in YEARS}

    summary_row([
        (f"Population {Y1}", f"{tot1 / 1000:.2f} million", f"{pct(tot0, tot1):+.1f}% vs {Y0}", "Sum of the selected states"),
        ("Fastest-growing state", short(p_g.idxmax()), f"{p_g.max():+.1f}%", "Highest % population growth 2022→2024"),
        (f"Most populous ({Y1})", short(pp[Y1].idxmax()), f"{pp[Y1].max() / 1000:.2f} million", None, "off"),
        (f"Aged 65+ in {Y1}", f"{old_share[Y1]:.1f}%", f"{old_share[Y1] - old_share[Y0]:+.1f} pts vs {Y0}", "Share of population aged 65 and above", "inverse"),
    ])

    c1, c2 = st.columns([3, 2])
    with c1:
        m = st_tab[st_tab.year == year].merge(
            p_all[p_all.year == year][["state", "population"]], on="state")
        m["pop_m"] = m["population"] / 1000
        m["lat"] = m["state"].map(lambda s: STATE_XY[s][0])
        m["lon"] = m["state"].map(lambda s: STATE_XY[s][1])
        fig = px.scatter_map(m, lat="lat", lon="lon", size="pop_m", color="income_median", hover_name="state",
                             hover_data={"pop_m": ":.2f", "income_median": ":,.0f", "lat": False, "lon": False},
                             size_max=48, zoom=4.4, center=dict(lat=4.3, lon=109.5),
                             color_continuous_scale="Viridis", map_style="carto-positron",
                             title=f"Chart 7 · Population (bubble size) and median income (colour), {year}",
                             labels={"pop_m": "Population (million)", "income_median": "Median income (RM)"})
        st.plotly_chart(style(fig, 470), width='stretch')
    with c2:
        scope = st.selectbox("Population pyramid for", ["All selected states"] + list(chosen))
        pyr = pop[(pop.year == year) & (pop.ethnicity == "overall") & (pop.sex.isin(["male", "female"])) &
                  (pop.age.isin(AGE_ORDER))]
        if scope != "All selected states":
            pyr = pyr[pyr.state == scope]
        pyr = pyr.groupby(["age", "sex"])["population"].sum().unstack().reindex(AGE_ORDER)
        fig = go.Figure()
        fig.add_bar(y=AGE_ORDER, x=-pyr["male"], orientation="h", name="Male", marker_color="#4C78A8",
                    customdata=pyr["male"], hovertemplate="Male %{y}: %{customdata:,.1f}k<extra></extra>")
        fig.add_bar(y=AGE_ORDER, x=pyr["female"], orientation="h", name="Female", marker_color="#E45756",
                    hovertemplate="Female %{y}: %{x:,.1f}k<extra></extra>")
        lim = float(max(pyr["male"].max(), pyr["female"].max()))
        step = lim / 2
        fig.update_layout(barmode="relative", bargap=0.05,
                          title=f"Chart 8 · Population pyramid, {year} ('000)",
                          xaxis=dict(title="Population ('000)", tickvals=[-lim, -step, 0, step, lim],
                                     ticktext=[f"{lim:,.0f}", f"{step:,.0f}", "0", f"{step:,.0f}", f"{lim:,.0f}"]),
                          yaxis_title="Age group")
        st.plotly_chart(style(fig, 470), width='stretch')

    inc_g = growth(st_tab, "income_median")
    gd = pd.concat([p_g.rename("pop_growth"), inc_g.rename("inc_growth"), pp[Y1].rename("pop")], axis=1).dropna().reset_index()
    gd["pop_m"] = gd["pop"] / 1000
    fig = px.scatter(gd, x="pop_growth", y="inc_growth", size="pop_m", color="state", text="state",
                     size_max=45, title="Chart 9 · Population growth vs median-income growth by state, 2022→2024",
                     labels={"pop_growth": "Population growth (%)", "inc_growth": "Median income growth (%)",
                             "pop_m": "Population 2024 (million)"})
    fig.update_traces(textposition="top center")
    fig.update_layout(showlegend=False)
    r9 = gd["pop_growth"].corr(gd["inc_growth"]) if len(gd) > 2 else float("nan")
    if len(gd) > 2:
        k9, c9 = np.polyfit(gd.pop_growth, gd.inc_growth, 1)
        xs = np.array([gd.pop_growth.min(), gd.pop_growth.max()])
        fig.add_trace(go.Scatter(x=xs, y=k9 * xs + c9, mode="lines", line=dict(color="black", dash="dash"),
                                 name="Trend", hoverinfo="skip"))
    st.plotly_chart(style(fig, 470), width='stretch')

    mx = m.loc[m.income_median.idxmax()]
    big = m.loc[m.population.idxmax()]
    mn = m.loc[m.income_median.idxmin()]
    shrink = p_g[p_g < 0].sort_values()
    interpretation([
        f"<b>Chart 7:</b> In {year}, {big.state} has the largest bubble ({big.pop_m:.2f} million people). "
        f"{mx.state} has the highest average median income (RM {mx.income_median:,.0f}) and {mn.state} the lowest "
        f"(RM {mn.income_median:,.0f}). Bright (high-income) bubbles are mostly on the west coast of Peninsular "
        f"Malaysia, while Sabah, Sarawak and the east coast are mostly darker.",
        f"<b>Chart 8:</b> The pyramid ({scope.lower() if scope != 'All selected states' else 'selected states'}, {year}) "
        f"is widest in the working-age groups. Between {Y0} and {Y1} the working-age share (15–64) moved from "
        f"{work_share[Y0]:.1f}% to {work_share[Y1]:.1f}% and the 65+ share from {old_share[Y0]:.1f}% to "
        f"{old_share[Y1]:.1f}%.",
        f"<b>Chart 9:</b> The correlation between population growth and income growth is r = {r9:.2f}. "
        f"{'Faster-growing states tend to also see faster income growth' if r9 > 0.3 else 'The link is weak, so population growth alone does not explain income growth'}. "
        f"{gd.loc[gd.inc_growth.idxmax(), 'state']} leads on income growth. "
        + (("Anomaly: population fell in " + ", ".join(f"{k} ({v:+.1f}%)" for k, v in shrink.items()) + ".")
           if len(shrink) else "No selected state lost population."),
    ], "Population and income are geographically concentrated around urban centres where jobs and industry are. "
       "A shift in the age structure towards older groups (population ageing) increases the dependency burden and "
       "can slow income growth in the long run. Correlation between growth rates does not prove that one causes the other.")

st.markdown("---")
st.markdown('<p class="small">Data: household income, household expenditure (by parliamentary seat) and population '
            '(by state), 2022 and 2024. Built with Streamlit and Plotly.</p>', unsafe_allow_html=True)
