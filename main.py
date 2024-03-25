import os
import datetime
from langchain.agents.agent_types import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from langchain_community.callbacks import get_openai_callback

import json
import flask
import pandas as pd

from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool

from langchain.agents.openai_functions_agent.base import (
    OpenAIFunctionsAgent,
    create_openai_functions_agent,
)

from dash import Dash, html, dcc
import plotly as plotly
import plotly.express as px
import numpy as np


#llm = ChatGoogleGenerativeAI(model="gemini-pro",verbose = True,temperature = 0.1,google_api_key="AIzaSyBvVehap53dXxJXqSu2ugGXYoWgc1ZT5QY",)
llm = ChatOpenAI(verbose = True,temperature = 0.5,openai_api_key="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")

df = pd.read_csv(
#    "./sales_report.csv",
#    "./sales_data_sample.csv", encoding='Latin-1',
#    "./metadata.csv", encoding='Latin-1',
    "./Amazon_Sale_Report.csv",
#    "./May_2022.csv",
#    "./airbnb.csv",
)

df = df.rename(columns=lambda x: x.strip().replace(' ', '_'))
df.convert_dtypes()
columns = df.columns
python = PythonAstREPLTool(locals={"df": df, "px": px, 'plt':plotly, 'pd':pd, 'np':np}) # set access of python_repl tool to the dataframe
dtypes = df.dtypes
head_data = df.head()
summary = df.describe()


@tool('DuckDuckGoSearch')
def search(search_query: str):
    """Search the web for information on a given topic"""
    return DuckDuckGoSearchRun().run(search_query)

personas = "Business owner"


@tool('CommandVerification')
def command_verification(command: str) -> str:
    """Verify df command for validity, run it for every command you created"""
    print("verifying " + command)
    try:
      python.run(command)
    except Exception as e:
      raise e
    return "Valid command"

def df_modification(json_str: str) -> str:
    """input should look like this {json:{"field_name": "datetime64"}}"""
    data = json.loads(json_str)
    for key, value in data.items():
      print("bbb")
      col = str(key)
      dtype = str(value)
      print("key: {0} | value: {1}".format(col, dtype))
      #if (col == 'DATE'):
      df[col] = df[col].astype(dtype)
    return df


personas = "Sales"


researcher = Agent(
  role='metrics_creator',
  goal=f'Create metrics which are important for the \"{personas}\" based on df.columns we have {columns} and provide 2 abnormalities detection metrics, propose insights, outliers metrics',
  backstory="You're professional data researcher",
  verbose=True,
  allow_delegation=False,
  llm = llm,
  #tools=[search]
)

dataframe_query_provider = Agent(
  role='data_analyst',
  goal=f'Provide information which metrics_creator requested from the dataframe based on df.dtypes we have {dtypes}',
  backstory="You're professional Data Analyst you know how to use pandas on advanced level, you get task from information from metrics_creator",
  verbose=True,
  allow_delegation=False,
  llm = llm,
  #tools=[df_modification]
)


task_datatypes = Task(description= f'Based on \"{head_data}\" and {dtypes} modify dataframe using Dataframe datatype modification prode json string as input for the tool',
  agent = dataframe_query_provider,
  expected_output=''
)

print(dtypes)
#task_datatypes.execute()
#df = df_modification(task_datatypes.execute())
print(df.dtypes)

plotly_expert = Agent(
  role='plotly_expert',
  goal=f'Provide plotly.express as px command based on the json you got from data_analyst',
  backstory="You're professional Data visualiser you are an expert in plotly.express",
  verbose=True,
  allow_delegation=False,
  llm = llm,
  #tools=[df_modification, command_verification]
)


task_research = Task(description= f'Give me 5 dataframe command list for you most valuable metrics IMPORTANT Make sure that all the columns exist, columns types are ({dtypes})',
  agent = researcher,
  expected_output='5 most interesting metrics discovered based on the data we have'
  )

task_json = Task(description= f'Give me 5 dataframe command list for you most valuable metrics IMPORTANT Make sure that all the columns exist, columns types are ({dtypes})',
  agent = plotly_expert,
  expected_output='Result of your research in json format like this ' +
  json.dumps({"metrics":
    [{'name':'Average Price per Property Type',
      'command':"df.groupby('Property Type')['Price'].mean()",
      'figure': "px.bar(df.groupby('Property Type')['Price'].mean().reset_index(), title='Average Price per Property Type', x='Property Type', y='Price')"},
     {'name':'Average Review Scores Rating per Neighbourhood',
      'command':"df.groupby('Neighbourhood')['Review Scores Rating'].mean()",
      'figure': "px.bar(df.groupby('Neighbourhood')['Review Scores Rating'].mean().reset_index(), title='Average Review Scores Rating per Neighbourhood', x='Neighbourhood', y='Review Scores Rating')"}]}
      )
  )

crew = Crew(
  agents=[researcher, dataframe_query_provider, plotly_expert],
  tasks=[task_research, task_json],
  verbose=1,
  process=Process.sequential
  #manager_llm=llm,  # Mandatory for hierarchical process
  #process=Process.hierarchical,
)

figures = []


server = flask.Flask('app')
app = Dash('app', server=server)

with get_openai_callback() as cb:
  result = crew.kickoff()
  result = json.loads(result)
  for metric in result['metrics']:
    print('')
    print(metric['name'])
    result = python.run(metric['command'])
    figure = python.run(metric['figure'])
    figures.append(
      dcc.Graph(
        figure=figure
      )
    )
    print(metric['figure'])
    print(result)


app.layout = html.Div(children=figures)
app.run_server(debug=True)

