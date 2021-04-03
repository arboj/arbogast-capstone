#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 18:34:55 2021

@author: Arbo

"""


from capstone_twitter_search import twittsearch
import os
import pandas as pd
import folium

# =============================================================================
# 
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# 
# import plotly.express as px
# =============================================================================



code_dir = os.getcwd()
print("Current working directory: {0}".format(code_dir))
parent_dir = os.path.dirname(code_dir)
data_dir = os.path.join(parent_dir,"Data")
tweet_dir = os.path.join(parent_dir,"TweetMap")

searchterms = ['snow','weather','power','freeze', 'ice', 'blackout','water']
text_query = "snow OR weather OR power OR freeze OR ice OR blackout OR water'"


since_date = '2021-02-12'
until_date = '2021-02-20'

listdfs = twittsearch(text_query,since_date,until_date)

tweets_geo_df =listdfs[0]
# =============================================================================
# tweets_geo_df = twittsearch(text_query,since_date,until_date)
# =============================================================================
tweets_no_geo_df = listdfs[1]

tweets_geo_df.to_csv(os.path.join(tweet_dir,"tweets_geo.csv"))
tweets_no_geo_df.to_csv(os.path.join(data_dir,"tweets_no_geo.csv"))



tweet_map = pd.read_csv(os.path.join(data_dir,"tweets_geo.csv"))



# =============================================================================
# fig = px.scatter_mapbox(tweet_map, lat="lat", lon="lon", hover_name="FoundWord", hover_data=["place_name", "Text", "Datetime"],
#                         color_discrete_sequence=["fuchsia"],title="sample figure", zoom=3, height=500)
# fig.update_layout(mapbox_style="open-street-map")
# fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# fig.show()
# 
# 
# app = dash.Dash()
# app.layout = html.Div([
#     dcc.Graph(figure=fig)
# ])
# 
# app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter
# =============================================================================


