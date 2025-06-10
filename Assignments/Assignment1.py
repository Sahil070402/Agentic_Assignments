from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Dict
import os


# Load environment variables
from dotenv import load_dotenv
load_dotenv()



# Initialize the model
model = ChatOpenAI(
    model="gpt-4o-2024-08-06",
    openai_api_key=os.getenv("OPEN_API_KEY"),
    openai_api_base=os.getenv("BASE_URL"),
)



# Define structured output model
class Product(BaseModel):
    product_name: str = Field(..., description="Name of the product")
    product_details: Dict[str, str] = Field(..., description="Key-value pairs describing product specifications")
    product_price: int = Field(..., description="Tentative price of the product in USD")


# Output parser
parser = PydanticOutputParser(pydantic_object=Product)
format_instructions = parser.get_format_instructions()


# Prompt message
human_message = HumanMessagePromptTemplate.from_template(
    "Give the details of the ecommerce product '{product}' and return the product name, a dictionary of its specifications, and its tentative price in USD.\n{format_instruction}",
    partial_variables={"format_instruction": format_instructions}
)


# Complete prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful ecommerce AI assistant. helping customers to give product data."),
    human_message
])

# Chain
chain = prompt | model | parser

# Run query
result = chain.invoke({'product': "Samsung S24"})
print(result)