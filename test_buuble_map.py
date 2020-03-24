import plotly.graph_objects as go
#df = px.data.gapminder().query("year==2007")
fig = go.Figure()

fig.add_trace(go.Scattergeo(
    lon=[21.0936],
    lat=[7.1881],
    text=['Africa'],
    mode='text',
    showlegend=False,
    geo='geo2'
))

fig.show()
