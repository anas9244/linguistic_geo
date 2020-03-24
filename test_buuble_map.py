# import plotly.graph_objects as go

# import pandas as pd

# fig = go.Figure()

cities_text = ['Lewisville, TX', 'Santa Clarita, CA',
               'Spokane, WA', 'Riverview, FL']


cities_loc = [(33.035303, -96.988046), (34.403326, -118.527593),
              (47.668131, -117.398425), (27.866140, -82.326241)]


# for i in range(len(cities_text)):

#     fig.add_trace(go.Scattergeo(
#         locationmode='USA-states',
#         # lon=(cities_loc[i][1], cities_loc[i][1] + 0.1),
#         # lat=(cities_loc[i][0], cities_loc[i][0] + 0.1),
#         lon=cities_loc[i][1],
#         lat=cities_loc[i][0],
#         text=cities_text[i],
#         marker=dict(
#             size=10,
#             line_color='rgb(40,40,40)',
#             line_width=0.5,
#             sizemode='area'
#         )))

# fig.update_layout(
#     title_text='2014 US city populations<br>(Click legend to toggle traces)',
#     showlegend=True,
#     geo=dict(
#         scope='usa',
#         landcolor='rgb(217, 217, 217)',
#     )
# )

# fig.show()


# # df = pd.read_csv(
# #     'https://raw.githubusercontent.com/plotly/datasets/master/2014_us_cities.csv')
# # df.head()

# # df['text'] = df['name'] + '<br>Population ' + \
# #     (df['pop'] / 1e6).astype(str) + ' million'
# # limits = [(0, 2), (3, 10), (11, 20), (21, 50), (50, 3000)]
# # colors = ["royalblue", "crimson", "lightseagreen", "orange", "lightgrey"]
# # cities = []
# # scale = 5000

# # fig = go.Figure()

# # for i in range(len(limits)):
# #     lim = limits[i]
# #     df_sub = df[lim[0]:lim[1]]
# #     fig.add_trace(go.Scattergeo(
# #         locationmode='USA-states',
# #         lon=df_sub['lon'],
# #         lat=df_sub['lat'],
# #         text=df_sub['text'],
# #         marker=dict(
# #             size=df_sub['pop'] / scale,
# #             color=colors[i],
# #             line_color='rgb(40,40,40)',
# #             line_width=0.5,
# #             sizemode='area'
# #         ),
# #         name='{0} - {1}'.format(lim[0], lim[1])))

# # fig.update_layout(
# #     title_text='2014 US city populations<br>(Click legend to toggle traces)',
# #     showlegend=True,
# #     geo=dict(
# #         scope='usa',
# #         landcolor='rgb(217, 217, 217)',
# #     )
# # )

# # fig.show()


import plotly.graph_objects as go

import pandas as pd


cities_text = ['Lewisville, TX', 'Santa Clarita, CA',
               'Spokane, WA', 'Riverview, FL']


cities_loc = [(33.035303, -96.988046), (34.403326, -118.527593),
              (47.668131, -117.398425), (27.866140, -82.326241)]
# df = pd.read_csv(
#     'https://raw.githubusercontent.com/plotly/datasets/master/2011_february_us_airport_traffic.csv')
# df['text'] = df['airport'] + '' + df['city'] + ', ' + \
#     df['state'] + '' + 'Arrivals: ' + df['cnt'].astype(str)

fig = go.Figure(data=go.Scattergeo(
    lon=[lon[1] for lon in cities_loc],
    lat=[lon[0] for lon in cities_loc],
    text=cities_text,
    mode='markers+text', textposition='bottom center'

))

fig.update_layout(
    title='Most trafficked US airports<br>(Hover for airport names)',
    geo_scope='usa',
)


fig.update_layout(

    geo=dict(
        scope='usa',
        landcolor='rgb(217, 217, 217)',
    )
)

fig.update_traces(marker=dict(size=120,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  textfont=dict(
    family="sans serif",
    size=71
))
fig.show()
