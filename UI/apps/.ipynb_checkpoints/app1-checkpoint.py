import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input,Output, State
import dash_bootstrap_components as dbc
from app import app
import functions
from flask_caching import Cache
cache = Cache(app.server, config={'CACHE_TYPE': 'simple'})

card2 = [
    dbc.CardBody([html.H1('BCI for Image Retrival', style = {'textAlign':'center', 'color':'black',}),
        
    ])
]
card3 = [
    dbc.CardBody([html.H3('Select a shape and a color', style = {'textAlign':'center', 'color':'black',}),
        
    ])
]
card4 = [
    dbc.Row([
    dbc.Col([html.P('Shape :', style = {'color':'black'})],  width = '650px', style = {'margin-left':215}),
    dbc.Col(dcc.Dropdown(
        id = 'shape_dd',
        options = [
            {'label':'Circle','value':'Circle'},
            {'label':'Heart','value':'Heart'},
            {'label':'Rhombus','value':'Rhombus'},
            {'label':'Square','value':'Square'},
            {'label':'Star','value':'Star'},
            {'label':'Triangle','value':'Triangle'},
        ],
        placeholder = 'Select shape',
        value = 'Circle',
        clearable = True,
        style = {'width':150}
    ), width = 2),
    dbc.Col([html.P('Color :')], width = '650px', style = {'margin-left':50}),
    dbc.Col(dcc.Dropdown(
        id = 'color_dd',
        options = [
            {'label':'Red','value':'Red'},
            {'label':'Green','value':'Green'},
            {'label':'Blue','value':'Blue'},
                
        ],
        placeholder = 'Select color',
        value = 'Green',
        clearable = True,
        style = {'width':150}
    ), width = 3),
    ], style = {'margin-top':25}),
]

card1 = [
    dbc.CardBody([
        dbc.CardLink("Plot", id = 'plot_a', n_clicks = 0, href="/app/app2", style = {'color':'black', 'textAlign':'center'}),
    ])
]

layout = html.Div([
    html.Div([dbc.Card(card2, color = 'pink')]),
    html.Div([dbc.Card(card3, color = 'pink')], style = {'width':600, 'margin-top':50, 'margin-left':480}),
    html.Div([dbc.Card(card4, color = 'pink', style = {'textAlign':'center'})], style = {'width':900, 'margin-top':50, 'margin-left':350}),
    #dcc.Store(id='s', data = [{'shape':'Circle', 'color':'Green'}]),
    html.H1(id = 'hidden'),
    html.Div([dbc.Card(card1, color = 'pink', outline = True, style = {'textAlign':'center', 'fontSize':35})],style = {'width':200, 'margin-left':680, 'margin-top':275}),
    
],

style = {
    'backgroundColor': 'white',
    'backgroundSize' : '100% 100%',
    'position': 'fixed',
    'min-height': '100%',
    'min-width': '100%',
}    
    
)

@app.callback(Output('hidden', 'children'),
    Input('plot_a', 'n_clicks'),
    [State('shape_dd', 'value'),
    State('color_dd', 'value')], 
    )
def update_output(n, shape_dd, color_dd):
    print("123")
    if(n != 0):
        print("Plot agthide")
        functions.plot(shape_dd, color_dd)
        functions.reconstruct()
