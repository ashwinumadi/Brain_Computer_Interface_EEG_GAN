import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import os
import base64
import time

from app import app
from apps import app1



cwd = os.getcwd()

app.layout = html.Div([
    dcc.Location(id = 'url', refresh = False),
    html.Div(id = 'page-content')
])

card = [
    dbc.CardBody([
        dbc.Col([
            dbc.Row([html.P("By-")]),
            dbc.Row([html.P("\tKushal N")]),
            dbc.Row([html.P("\tAshwin Umadi")]),
            dbc.Row([html.P("\tNagavishnu B K")]),
        ])
    ], style = { 'color':'black'})
]

card1 = [
    dbc.CardBody([
        dbc.CardLink("Start", href="/app/app1", style = {'color':'black', 'textAlign':'center'}),
    ])
]
card2 = [
    dbc.CardBody([html.H1('BCI for Image Retrival', style = {'textAlign':'center', 'color':'black',}),
        
    ])
]

index_page = html.Div([
    html.Div([dbc.Card(card2, color = 'pink')]),
    dbc.Row([
        dbc.Col([html.Div([dbc.Card(card1, color = 'pink', outline = True, style = {'textAlign':'center', 'fontSize':35})],style = {'width':200}),],width = "100px", style = {'margin-left':690, 'margin-top':150}),
        dbc.Col([html.Div([dbc.Card(card, color = 'pink')],style = {'width':300}),],width = "100px", style = {'margin-left':400, 'margin-bottom':50}),
        
    ],
    style = {'margin-top':400})
   
    
],
style = {
    'backgroundImage': 'url("./assets/eegsignals.jpg")',
    'backgroundSize' : '100% 100%',
    'position': 'fixed',
    'min-height': '100%',
    'min-width': '100%',
}
)

@app.callback(Output('page-content','children'),
            Input('url','pathname'))
def display_page(pathname):
    if pathname == '/app/app1':
        #time.sleep(5)
        return app1.layout
    elif pathname == '/app/app2':
        time.sleep(10)
        from apps import app2
        return app2.layout1
    elif pathname == '/app/app3':
        #time.sleep(10)
        from apps import app3
        return app3.layout2
    elif pathname == '/':
        return index_page
    else:
        return '404'
    
if __name__ == '__main__':
    app.run_server(debug = True)