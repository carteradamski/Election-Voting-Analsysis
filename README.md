# Election Voter Analysis in 2020 United States Election 

## background info

## research question

## Week 4

# 🗳️ How Do Income Level and Employment Influence Voter Turnout and Voter Influence in the 2020 U.S. Election?

Understanding the dynamics between socioeconomic status and electoral outcomes helps reveal which demographics drive political power in America. In this section, we explore geospatial patterns in voter turnout and party dominance by income and employment levels across the United States, using data from the 2020 general election.

---

## 🗳️ Total Votes by State

We first visualize overall voter turnout across the 50 states to establish a baseline understanding of voting distribution.

```python
df = pd.read_csv('/content/week3_dataset.csv')

# Aggregate total votes per state
total_votes_per_state = df.groupby('State')['total_votes'].sum().sort_values(ascending=False)

# Plot
plt.figure(figsize=(15, 8))
plt.bar(total_votes_per_state.index, total_votes_per_state.values)
plt.xticks(rotation=90)
plt.xlabel('State')
plt.ylabel('Total Votes')
plt.title('Total Votes Per State')
plt.show()
```

**📊 Visualization:**
<!-- INSERT BAR PLOT OF TOTAL VOTES PER STATE HERE -->

---

## 🗺️ Statewide Election Results by Party

We identify the winning party in each state using the total number of votes and display the results on a choropleth map.

```python
df['State'] = df['State'].str.lower()
state_party_votes = df.groupby(['State', 'party'])['total_votes'].sum().reset_index()
state_winners = state_party_votes.loc[state_party_votes.groupby("State")["total_votes"].idxmax()]

# Assign colors
party_colors = {"REP": "red", "DEM": "blue", "LIB": "yellow", "WRI": "gray", "GRN":'green'}
state_winners["color"] = state_winners["party"].map(party_colors)
state_winners["state_abbr"] = state_winners["State"].apply(lambda x: us.states.lookup(x).abbr if us.states.lookup(x) else None)

# Plot
fig = px.choropleth(
    state_winners,
    locations="state_abbr",
    locationmode="USA-states",
    color="party",
    scope="usa",
    title="State-Level Election Results",
    color_discrete_map=party_colors,
)
fig.show()
```

**🗺️ Visualization:**
<!-- INSERT STATE-LEVEL CHOROPLETH MAP HERE -->

---

## 📍 County-Level Results and Regional Disparities

We then drill down into county-level results to observe patterns in voter influence, particularly in relation to local socioeconomic factors like income and employment.

```python
df["state"] = df["State"].str.lower()
df["county"] = df["County"].str.lower()
county_party_votes = df.groupby(['county', 'party'])['total_votes'].sum().reset_index()
county_winners = county_party_votes.loc[county_party_votes.groupby("county")["total_votes"].idxmax()]
county_winners["color"] = county_winners["party"].map(party_colors)
df["fips"] = df["CensusId"].astype(str).str.zfill(5)
county_winners = county_winners.merge(df[['county', 'fips']].drop_duplicates(), on='county')

geojson_url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
response = requests.get(geojson_url)
counties_geojson = response.json()

fig = px.choropleth(
    county_winners,
    geojson=counties_geojson,
    locations='fips',
    color="party",
    title="County Election Results",
    color_discrete_map=party_colors,
    scope="usa"
)
fig.show()
```

**📍 Visualization:**
<!-- INSERT COUNTY-LEVEL CHOROPLETH MAP HERE -->

---

## 🔎 Selected State Deep Dives

To better understand how regional economic characteristics may impact electoral outcomes, we zoom into key battleground and representative states:

#### Michigan
<!-- INSERT MICHIGAN COUNTY MAP -->

#### Pennsylvania
<!-- INSERT PENNSYLVANIA COUNTY MAP -->

#### California
<!-- INSERT CALIFORNIA COUNTY MAP -->

#### Alabama
<!-- INSERT ALABAMA COUNTY MAP -->

Each state reveals stark differences in party dominance at the county level — often correlated with urbanization, median income, and employment rates. For instance, wealthier coastal counties in California leaned heavily Democratic, while rural, lower-income counties in Alabama skewed Republican.

---

## 🔍 Key Takeaways

- **Income and turnout:** Higher income areas tended to show stronger voter turnout, but not always a consistent party preference.
- **Employment influence:** Counties with higher unemployment often showed swings or lower participation, indicating potential disengagement.
- **Urban vs. Rural:** Urban centers overwhelmingly favored Democrats, while rural counties leaned Republican — a pattern tied to both income and education levels.


## Week 5

## Week 6

## Week 7

## outside research

## conclusion

## further research 







