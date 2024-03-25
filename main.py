import os
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from langchain.callbacks import get_openai_callback

llm = ChatGoogleGenerativeAI(model="gemini-pro",verbose = True,temperature = 0.1,google_api_key="AIzaSyBvVehap53dXxJXqSu2ugGXYoWgc1ZT5QY")

from langchain.tools import DuckDuckGoSearchRun

from crewai_tools import tool

@tool('DuckDuckGoSearch')
def search(search_query: str):
    """Search the web for information on a given topic"""
    return DuckDuckGoSearchRun().run(search_query)

researcher = Agent(
  role='news_reporter',
  goal=f'You provide the latest news as of available till {datetime.datetime.now()}',
  backstory="You're an professional news reporter who covers tech related news and are the best in that field",
  verbose=True,
  allow_delegation=False,
  llm = llm,
  tools=[search]
)

task1 = Task(description= 'give me a detailed overview of the AI brands and products showcased at ces 2024 hosted in las vegas using only 2024 articles. Cover 10 topics with 5 paragraphs of text for each topic',agent = researcher,expected_output='A refined finalized version of the blog post in markdown format')

crew = Crew(
  agents=[researcher],
  tasks=[task1],
  verbose=2, 
  process=Process.sequential
)

# This counts the amount of Gemini API Requests completed by the script. This is helpful given the 60 API requests per minute limit from gemini pro free api.

with get_openai_callback() as cb:
  result = crew.kickoff()
  print(result)
  print(cb)