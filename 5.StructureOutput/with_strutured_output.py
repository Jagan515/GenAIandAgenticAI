from ctypes import Structure
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from typing import TypedDict,Annotated,Optional,Literal
from dotenv import load_dotenv


load_dotenv()
 # TypedDict helps in creating a schema

 #Annoted helps in understanding the keywords like summmary,sentiments


 #literals helps in setting like postive:pos or set negative:neg
llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceTB/SmolLM3-3B",
    task="text-generation",
    max_new_tokens=20
)

model=ChatHuggingFace(llm=llm)
#schema
class Review(TypedDict):
    key:Annotated[list[str],"Write the key themes written in the review"]
    summary:Annotated[str,"A brief summary of the review"]
    sentiment:Annotated[str,"Return sentiment of the view either positive, negative or neutral"]
    Pros:Annotated[Optional[list[str]],"Write the down the pros inside the list"]
    Cons:Annotated[Optional[list[str]],"Write the down the cons inside the list"]


Structure_model=model.with_structured_output(Review)
response=Structure_model.invoke("""Not only that, but Black Friday 2025 is fast approaching, and once again I expect to see loads of brilliant laptop deals throughout November, up until the day itself (and I'll be leading TechRadar's coverage of Black Friday laptop deals). So, many people will have been waiting to buy a new laptop in the hope that there's some big price cuts, and that means guides like this best laptops list are incredibly important, as we give you clear, honest buying advice to help you buy the best laptop for your needs, no matter what your budget is.

I've carefully hand-picked every laptop that's on this page, and each laptop has been throughly tested by the computing team at TechRadar (you can read about how we test laptops, and all the ways we put laptops through their paces during our reviews), so only the very best laptops have made their way onto this list.

That's why you won't find the latest MacBook Pro with M5 on this list. While it is a great laptop, it's not quite revolutionary enough to earn a position here. Instead, the 16-inch MacBook Pro with the M4 Pro is our choice for the best laptop for creatives thanks to its larger screen and more powerful hardware.

It's not all bad news for Apple, however, as our number one pick for the best laptop remains the Apple MacBook Air 13-inch (M4). Its excellent performance, premium design and long battery life has yet to be beaten, though Microsoft has come close with the Surface Laptop 13-inch. Having trouble choosing between the two? Then good news, as we have a dedicated article where we pit the MacBook Air (M4) against the Surface Laptop 13-inch to see which one is worthy of your hard-earned cash.

I constantly update this page to make sure it contains the very latest laptops, MacBooks and Chromebooks, as well as essential buying advice to help you pick the best laptop for your needs, so you can shop safe in the knowledge that any laptop we recommend on this page is well worth the money - especially if they get any Black Friday price cuts.

""")


# print(response['summary'])

print(response)


