"""Handle all ChatGPT requests. Answer questions about github descriptions and git issues seniment analysis."""
import githubmonitor.api.schemas
import openai
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from framework.common import llm
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from tools.nlp import nlp_description
from tools.sentiment import llm_sentiment

router = APIRouter()

# Router basic config
llm_router = APIRouter(
    prefix="/langchain",
    tags=[
        "langchain",
    ],
    responses={404: {"description": "User not found"}},
)

MEM_KEY = "chat_history"

memory = ConversationBufferMemory(
    memory_key=MEM_KEY,
    return_messages=True,
    input_key="input",
    output_key="output",
)

nlp_description_prompt = """Use this tool when a description of a Git repository is provided. To identify this type of queries correctly, use the following criteria.
        1. The query must start with *** at the beginning of the description.
        2. The description must be a compatible GitHub description.
        3. If the description is too short, elaborate over the provided text, increasing the temperature of this prompt and provide more detail or context.

        Your response should be of a maximum of 280 characters. Example:*** Implements python and React to create a forecast model that predicts the number of commits over the following 12 weeks. Use LangChain framework to summarize and describe important information about the repositories and create interactive plots with ChartJS.

        """

llm_sentiment_prompt = """Use this tool when a new query is passed that fulfills the following two requirements:
        1 The query must have the following structure {repository_issue_title} | {repository_issue_description}. Where repository_issue_title is the title of a Git issue and repository_issue_description is the description (more details) about that same issue.
        2. The text should match a GitHub repository issue format. If you identify content that doesn't fall in this category or that cannot be further analyzed, return with a string "Unprocessable Issue

        The response from this Tool can only be one of: fatal, important conflicts, noProblems, perfect.

        These responses categorize the importance of the issue. Use your criteria to allocate the issue in one and only one of these categories. Your response should be a single-word response, for example, fatal. The output is the string single word alone, no formatting, no breaks or spaces -> "fatal".
        """


bot_tools = [
    Tool(
        name="llm_sentiment",
        func=llm_sentiment,
        description=llm_sentiment_prompt,
        coroutine=llm_sentiment,
        return_direct=True,
    ),
    Tool(
        name="nlp_description",
        func=nlp_description,
        description=nlp_description_prompt,
        coroutine=nlp_description,
        return_direct=True,
    ),
]

system_msg = SystemMessage(
    content="""
    You work for Metalab. Metalab is the best development company that excels in creating interactive and immersive applications.
    You are a very skilled Developer, proficient in all programming languages.
    You provide answers that are brief and insightful.
    You are proficient in psychoanalyst and psychology. \n
    You can never respond to inappropriate or out-of-scope questions.\n

    You always use a 'tool' to find answers to questions.\n
    Your tools allow you to answer create summaries about repositories descriptions and sentiment analysis over GitHub issues.\n Nothing else.
    Don't make up information that is not directly quotable from sources. If you do not know the answer then apologize and ask the user if they want to talk to a customer representative.\n
    Some sample questions and their tools are:\n

    llm_sentiment: useful to process GitHub issues and classify them using a criteria of importance of the issue.
    nlp_description: useful to process GitHub repositories descriptions


    """
)
prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_msg,
    extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],
)

agent = OpenAIFunctionsAgent(llm=llm, prompt=prompt, tools=bot_tools)

agent_executor = AgentExecutor(
    agent=agent,
    tools=bot_tools,
    verbose=True,
    memory=memory,
    return_intermediate_steps=True,
)


@llm_router.post("/interact_llm")
async def interact_llm(
    message: githubmonitor.api.schemas.MessageSchema
) -> StreamingResponse:
    """Endpoint for interacting with the OpenAI model to generate a response based on user input.

    Args:
        message (githubmonitor.api.schemas.MessageSchema): The request body containing the user's query.

    Returns:
        StreamingResponse: A streaming response object with the generated response.
    """
    query = message.query
    # Extract relevant information from the agent
    prompt = agent.prompt
    tools = agent.tools

    input_data = f"{prompt} {tools} {query}"

    # Make the OpenAI API call using openai.Completion
    response = openai.Completion.create(
        model="text-davinci-003",  # Use the correct model name
        prompt=input_data,
        max_tokens=150,
    )

    # Extract the output from the response
    output = response["choices"][0]["text"]
    return {"response": output}


@llm_router.post("/nlp_description")
async def nlp_description(
    message: githubmonitor.api.schemas.MessageSchema
) -> StreamingResponse:
    """Endpoint for processing GitHub repository descriptions using the NLP tool.

    Args:
        message (githubmonitor.api.schemas.MessageSchema): The request body containing the repository description.

    Returns:
        JSONResponse: A JSON response object with the processed description.
    """
    query = message.query

    input_data = f"{nlp_description_prompt} with this tool answer this query: {query}"

    # Make the OpenAI API call using openai.Completion
    response = openai.Completion.create(
        model="text-davinci-003",  # Use the correct model name
        prompt=input_data,
        max_tokens=150,
    )

    # Extract the output from the response
    output = response["choices"][0]["text"]
    response = {"description": output}

    return JSONResponse(response)


@llm_router.post("/issue_sentiment")
async def issue_sentiment(
    message: githubmonitor.api.schemas.MessageSchema
) -> StreamingResponse:
    """Endpoint for processing GitHub issues and classifying them using the sentiment analysis tool.

    Args:
        message (githubmonitor.api.schemas.MessageSchema): The request body containing the GitHub issue details.

    Returns:
        JSONResponse: A JSON response object with the classified category.
    """
    query = message.query

    input_data = f"{llm_sentiment_prompt} with this context answer this query: {query}"

    # Make the OpenAI API call using openai.Completion
    response = openai.Completion.create(
        model="text-davinci-003",  # Use the correct model name
        prompt=input_data,
        max_tokens=150,
    )

    # Extract the output from the response
    output = response["choices"][0]["text"]
    output = (
        output.replace("\n", "")
        .replace(" ", "")
        .replace(",", "")
        .replace(".", "")
        .lower()
    )
    if output not in ["fatal", "important", "conflicts", "noProblems", "perfect"]:
        # Set default
        output = "noProblems"

    response = {"category": output}

    return JSONResponse(response)
