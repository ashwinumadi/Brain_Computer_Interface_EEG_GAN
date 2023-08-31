import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input,Output
import dash_bootstrap_components as dbc
from app import app
import functions
import base64
import os
from apps import app1

cwd = os.getcwd()

card2 = [
    dbc.CardBody([html.H1('BCI for Image Retrival', style = {'textAlign':'center', 'color':'black',}),
        
    ])
]

card1 = [
    dbc.CardBody([
        dbc.CardLink("Once More", id = 'plot_c', n_clicks = 0, href="/", style = {'color':'black', 'textAlign':'center'}),
    ])
]

file_name = "./flask1/app/repo/storage.txt"
f = open(file_name, 'r')
shapes = f.readline()
num = f.readline()
num = int(num)
num-=1
num = str(num)
f.close()
print("From app3")
string = "./assets/reconstructed" + num + ".png"
reconstructed = base64.b64encode(open(string, 'rb').read()).decode("ascii")
    
layout2 = html.Div([

    html.Div([dbc.Card(card2, color = 'pink')]),
    html.Br(),
    html.Img(src = "data:image/png;base64,{}".format(reconstructed), style = {'margin-left':'32%'}),
    html.Br(),
    html.Div([dbc.Card(card1, color = 'pink', outline = True, style = {'textAlign':'center', 'fontSize':35})],style = {'width':300, 'margin-left':680, 'margin-top':40}),
    
],

'''style = {
    'backgroundColor': 'black',
    'backgroundSize' : '100% 100%',
    'position': 'fixed',
    'min-height': '100%',
    'min-width': '100%',
}    '''
)


