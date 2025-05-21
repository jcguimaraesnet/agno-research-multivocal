# /// script
# dependencies = ["openai", "googlesearch-python", "pycountry", "agno", "duckduckgo-search", "newspaper4k", "lxml_html_clean", "beautifulsoup4", "requests", "firecrawl", ]
# [tool.uv]
# exclude-newer = "2025-05-01T00:00:00Z"
# ///

# uv venv --python 3.12
# .venv/Scripts/activate
# uv run .\main.py

import json
from textwrap import dedent
from typing import Dict, Iterator, Optional

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.utils.log import logger
from agno.utils.pprint import pprint_run_response
from agno.workflow import RunEvent, RunResponse, Workflow
from agno.tools import tool
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.tools.website import WebsiteTools
from agno.tools.firecrawl import FirecrawlTools
from googleapiclient.discovery import build

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from rich.prompt import Prompt
import os

load_dotenv()

@tool(
    description="Search the web using Google Search Engine",
    show_result=True,                               # Show result after function call
    stop_after_tool_call=False,                      # Return the result immediately after the tool call and stop the agent
    cache_results=True,                             # Enable caching of results
    cache_dir="/tmp/agno_cache",                    # Custom cache directory
    cache_ttl=3600                                  # Cache TTL in seconds (1 hour)
)
def google_search(query: str = "", num_results: int = 5) -> list[tuple[str, str]]:
    """
    Search the web for articles on the query

    Args:
        query: string - The query to search for
        num_results: int - The number of results to return

    Returns:
        A list of tuples, where each tuple contains the title and link of the article
    """
    try:
        # Obtém as credenciais do arquivo .env
        api_key = os.getenv("GOOGLE_API_KEY")
        search_engine_id = os.getenv("GOOGLE_CSE_ID")
        
        if not api_key or not search_engine_id:
            raise ValueError("API key ou Search Engine ID não encontrados no arquivo .env")

        # Cria o serviço de busca
        service = build("customsearch", "v1", developerKey=api_key)

        # Realiza a busca
        result = service.cse().list(
            q=query,
            cx=search_engine_id,
            num=num_results
        ).execute()

        # Extrai os resultados
        items = result.get("items", [])
        return [(item["title"], item["link"]) for item in items]

    except Exception as e:
        print(f"Error searching the web: {str(e)}")
        return []


class NewsArticle(BaseModel):
    title: str = Field(..., description="Title of the article.")
    link: str = Field(..., description="Link to the article.")
    summary: Optional[str] = Field(
        description="Summary of the article if available."
    )

class SearchResults(BaseModel):
    articles: list[NewsArticle]

class ScrapedArticle(BaseModel):
    source_id: str = Field(description="Calculated ID sequence starting with A1, A2, A3, etc.")
    title: str = Field(description="Title of the article.")
    link: str = Field(description="Link (url) to the article.")
    authors: Optional[str] = Field(description="Author(s) of the article if available.")
    abstract: Optional[str] = Field(description="Summary of the article if available.")
    year: int = Field(description="Year of the publication of the article.")
    publication_venue: Optional[str] = Field(description="Publication venue of the article if available.")
    country_authors: Optional[str] = Field(description="Country of the authors of the article if available.")
    content: Optional[str] = Field(description="Content of the article if available.")

class ResearchQuestions(BaseModel):
    scraped_article: ScrapedArticle
    rq1: Optional[str] = Field(description="Which Generative Artificial Intelligence-based solutions (tools, agents, models, etc.) are available to support Software Engineering activities, considering the contexts of industry and academia?")
    rqa: Optional[str] = Field(description="Where can these solutions be found?")
    rqb: Optional[str] = Field(description="How can these solutions be used?")
    rqc: Optional[str] = Field(description="What is the access and usage model for these solutions and costs?")
    rqd: Optional[str] = Field(description="What limitations must be considered for adopting these solutions?")
    rqe: Optional[str] = Field(description="What is the maturity level of these solutions?")
    rqf: Optional[str] = Field(description="Is there technical documentation, support, or an active community to facilitate these solutions in academia and industry?")
    rqg: Optional[str] = Field(description="What types of target users or roles typically engage with the proposed solutions? (e.g., requirements engineers, testers, developers)")
    rqh: Optional[str] = Field(description="Is there any collaboration with the industry to validate the proposals through case studies or experiments?")

class InclusionCriterias(BaseModel):
    scraped_article: ScrapedArticle
    IC1: bool = Field(description="If the article is the context of Software Engineering, return True, otherwise return False.")
    IC1_reason: str = Field(description="Reason for the inclusion criterias 1.")
    IC2: bool = Field(description="If the article present a solution generative AI applicable to Software Construction, return True, otherwise return False.")
    IC2_reason: str = Field(description="Reason for the inclusion criterias 2.")
    IC3: bool = Field(description="If the article report a primary study or a case of application in the field, return True, otherwise return False.")
    IC3_reason: str = Field(description="Reason for the inclusion criterias 3.")
    IC4: bool = Field(description="If the article report a deeployed solution that can be used in the industry, return True, otherwise return False.")
    IC4_reason: str = Field(description="Reason for the inclusion criterias 4.")
    IC5: bool = Field(description="If the article has beed published after December 2021.")
    IC5_reason: str = Field(description="Reason for the inclusion criterias 5.")
    IC6: bool = Field(description="If the article provide data to answer at least one of the research questions (rq1, rqa, rqb, rqc, rqd, rqe, rqf, rqg, rqh), return True, otherwise return False.")
    IC6_reason: str = Field(description="Reason for the inclusion criterias 6.")

class MultiAgentResearch(Workflow):
    """Advanced workflow for selection article for research."""
    description: str = dedent("""\
    An intelligent blog post generator that creates engaging, well-researched content.
    This workflow orchestrates multiple AI agents to research, analyze, and craft
    compelling blog posts that combine journalistic rigor with engaging storytelling.
    The system excels at creating content that is both informative and optimized for
    digital consumption.
    """)

    # Search Agent: Handles intelligent web searching and source gathering
    searcher: Agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        # tools=[GoogleSearchTools()],
        tools=[google_search],
        description=dedent("""\
        Search the web using Google Search Engine
        """),
        instructions=dedent("""\
        Given a topic by the user, respond with 2 results of items about that topic.
        """),
        response_model=SearchResults,
        show_tool_calls=True,
        debug_mode=False,
    )

    # Content Scraper: Extracts and processes article content
    article_scraper: Agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[FirecrawlTools(scrape=True, crawl=False)],
        description=dedent("""\
        You are ContentBot-X, a specialist in Scrape a Website for blog creation. 

        Your expertise includes:
        - Efficient content extraction
        - Smart formatting and structuring
        - Key information identification
        - Quote and statistic preservation
        - Maintaining source attribution\
        """),
        instructions=dedent("""\
        1. Content Extraction 📑
           - Extract content from the article
           - Preserve important quotes and statistics
           - Maintain proper attribution
           - Handle paywalls gracefully
        2. Content Processing 🔄
           - Format text in clean markdown
           - Preserve key information
           - Structure content logically
        3. Quality Control ✅
           - Verify content relevance
           - Ensure accurate extraction
           - Maintain readability\
        """),
        response_model=ScrapedArticle,
        show_tool_calls=True,
        debug_mode=False
    )

    research_questions_agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        description=dedent("""\
            You are a software engineering researcher, skilled in 
            answering research questions based on an article's title, 
            abstract, and metadata.\
        """),
        instructions=dedent("""\
            Given a user-provided article, answer research questions about that article.\
        """),
        response_model=ResearchQuestions,
        show_tool_calls=True,
        debug_mode=False
    )

    research_inclusion_criterias_agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        description=dedent("""\
            You are a researcher that applys inclusion criterias to a article.\
        """),
        instructions=dedent("""\
            Apply the inclusion criterias to the article.
        """),
        response_model=InclusionCriterias,
        show_tool_calls=True,
        debug_mode=False
    )

    def get_cached_search_results(self, topic: str) -> Optional[SearchResults]:
        logger.info("Checking if cached search results exist")
        search_results = self.session_state.get("search_results", {}).get(topic)
        return (
            SearchResults.model_validate(search_results)
            if search_results and isinstance(search_results, dict)
            else search_results
        )

    def add_search_results_to_cache(self, topic: str, search_results: SearchResults):
        logger.info(f"Saving search results for topic: {topic}")
        self.session_state.setdefault("search_results", {})
        self.session_state["search_results"][topic] = search_results

    def get_cached_scraped_articles(
        self, topic: str
    ) -> Optional[Dict[str, ScrapedArticle]]:
        logger.info("Checking if cached scraped articles exist")
        scraped_articles = self.session_state.get("scraped_articles", {}).get(topic)
        return (
            ScrapedArticle.model_validate(scraped_articles)
            if scraped_articles and isinstance(scraped_articles, dict)
            else scraped_articles
        )

    def add_scraped_articles_to_cache(
        self, topic: str, scraped_articles: Dict[str, ScrapedArticle]
    ):
        logger.info(f"Saving scraped articles for topic: {topic}")
        self.session_state.setdefault("scraped_articles", {})
        self.session_state["scraped_articles"][topic] = scraped_articles

    def get_search_results(
        self, topic: str, use_search_cache: bool, num_attempts: int = 3
    ) -> Optional[SearchResults]:
        # Get cached search_results from the session state if use_search_cache is True
        if use_search_cache:
            try:
                search_results_from_cache = self.get_cached_search_results(topic)
                if search_results_from_cache is not None:
                    search_results = SearchResults.model_validate(
                        search_results_from_cache
                    )
                    logger.info(
                        f"Found {len(search_results.articles)} articles in cache."
                    )
                    return search_results
            except Exception as e:
                logger.warning(f"Could not read search results from cache: {e}")

        # If there are no cached search_results, use the searcher to find the latest articles
        for attempt in range(num_attempts):
            try:
                searcher_response: RunResponse = self.searcher.run(topic)
                pprint_run_response(searcher_response)
                if (
                    searcher_response is not None
                    and searcher_response.content is not None
                    and isinstance(searcher_response.content, SearchResults)
                ):
                    article_count = len(searcher_response.content.articles)
                    logger.info(
                        f"Found {article_count} articles on attempt {attempt + 1}"
                    )
                    # Cache the search results
                    self.add_search_results_to_cache(topic, searcher_response.content)
                    return searcher_response.content
                else:
                    logger.warning(
                        f"Attempt {attempt + 1}/{num_attempts} failed: Invalid response type"
                    )
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{num_attempts} failed: {str(e)}")

        logger.error(f"Failed to get search results after {num_attempts} attempts")
        return None

    def scrape_articles(
        self, topic: str, search_results: SearchResults, use_scrape_cache: bool
    ) -> Dict[str, ScrapedArticle]:
        scraped_articles: Dict[str, ScrapedArticle] = {}

        # Get cached scraped_articles from the session state if use_scrape_cache is True
        if use_scrape_cache:
            try:
                scraped_articles_from_cache = self.get_cached_scraped_articles(topic)
                if scraped_articles_from_cache is not None:
                    scraped_articles = scraped_articles_from_cache
                    logger.info(
                        f"Found {len(scraped_articles)} scraped articles in cache."
                    )
                    return scraped_articles
            except Exception as e:
                logger.warning(f"Could not read scraped articles from cache: {e}")

        # Scrape the articles that are not in the cache
        for article in search_results.articles:
            if article.link in scraped_articles:
                logger.info(f"Found scraped article in cache: {article.link}")
                continue

            run_response: RunResponse = self.article_scraper.run(
                article.link
            )
            if (
                run_response is not None
                and run_response.content is not None
                and isinstance(run_response.content, ScrapedArticle)
            ):
                pprint_run_response(run_response)
                # from rich.json import JSON
                # from agno.cli.console import console
                # single_response_content = JSON(run_response.content.model_dump_json(exclude_none=True), indent=2)
                # console.print(single_response_content)
                scraped_articles[run_response.content.link] = (
                    run_response.content
                )
                logger.info(f"Scraped article: {run_response.content.link}")

        # Save the scraped articles in the session state
        self.add_scraped_articles_to_cache(topic, scraped_articles)
        return scraped_articles

    def generate_research_questions(
        self,
        scraped_articles: Dict[str, ScrapedArticle]
    ) -> Dict[str, ResearchQuestions]:
        research_questions: Dict[str, ResearchQuestions] = {}

        # for article in scraped_articles:
        for article in scraped_articles.values():
            #logger.info(f"Generating research questions for article: {article.model_dump()}")
            research_questions_response: RunResponse = self.research_questions_agent.run(
                json.dumps(article.model_dump(), indent=4)
            )
            pprint_run_response(research_questions_response)
            if (
                research_questions_response is not None
                and research_questions_response.content is not None
                and isinstance(research_questions_response.content, ResearchQuestions)
            ):
                research_questions[research_questions_response.content.scraped_article.link] = (
                    research_questions_response.content
                )
                # logger.info(f"Research questions generated: {research_questions_response.content.link}")

        return research_questions

    def apply_inclusion_criterias(
        self, 
        research_questions_articles: Dict[str, ResearchQuestions]
    ) -> Dict[str, InclusionCriterias]:
        articles_with_inclusion_criterias: Dict[str, InclusionCriterias] = {}

        # Apply the inclusion criterias
        for article in research_questions_articles.values():
            inclusion_criterias_response: RunResponse = self.research_inclusion_criterias_agent.run(
                json.dumps(article.model_dump(), indent=4)
            )
            pprint_run_response(inclusion_criterias_response)
            if (
                inclusion_criterias_response is not None
                and inclusion_criterias_response.content is not None
                and isinstance(inclusion_criterias_response.content, InclusionCriterias)
            ):
                articles_with_inclusion_criterias[inclusion_criterias_response.content.scraped_article.link] = (
                    inclusion_criterias_response.content
                )
                # logger.info(f"Criteria applyed: {inclusion_criterias_response}")

        # Save the scraped articles in the session state
        # self.add_scraped_articles_to_cache(topic, scraped_articles)
        return articles_with_inclusion_criterias

    def run(
        self,
        topic: str,
        use_search_cache: bool = True,
        use_scrape_cache: bool = True,
    ) -> Iterator[RunResponse]:
        logger.info(f"Generating a blog post on: {topic}")

        # Search the web for articles on the topic
        search_results: Optional[SearchResults] = self.get_search_results(
            topic, use_search_cache
        )

        logger.info(f"Found {len(search_results.articles)} articles")

        # If no search_results are found for the topic, end the workflow
        if search_results is None or len(search_results.articles) == 0:
            logger.info("Não encontrei nenhum artigo!")
            # yield RunResponse(
            #     event=RunEvent.workflow_completed,
            #     content=f"Sorry, could not find any articles on the topic: {topic}",
            # )
            # return

        # Scrape the search results
        scraped_articles: Dict[str, ScrapedArticle] = self.scrape_articles(
            topic, search_results, use_scrape_cache
        )

        # Generate the research questions
        research_questions_articles: Dict[str, ResearchQuestions] = self.generate_research_questions(
            scraped_articles
        )

        # Apply the inclusion criterias
        articles_with_inclusion_criterias: Dict[str, InclusionCriterias] = self.apply_inclusion_criterias(
            research_questions_articles
        )

        yield RunResponse(
            content=articles_with_inclusion_criterias, event=RunEvent.workflow_completed
        )

# Run the workflow if the script is executed directly
if __name__ == "__main__":
    
    from rich.prompt import Prompt

    # Initialize the agent with the model and tools
    workflow = MultiAgentResearch()
    
    # Get the topic from the user
    topic = Prompt.ask(
        "[bold]Enter a blog post topic[/bold] (or press Enter for a default example)\n✨", 
        default="site:github.com awesome ai code generation assistant"
    )
    # default="(\"Code Generation\" OR \"Coding Generation\" OR \"Coding Assistant\" OR \"Code Assistant\" OR \"Software Development\") AND (\"Generative AI\" OR \"Generative Artificial Intelligence\" OR \"Generative Model*\" OR \"Large Language Model*\" OR \"Language Model*\" OR \"Small Language Model*\" OR \"LLM*\" OR \"RAG\" OR \"Retrieval Augmented Generation\" OR \"Natural Language Processing\" OR \"NLP\" OR \"AI Agent\" OR \"LLM-based agent\" OR \"AI Multi-Agent\") AND (\"Application\" OR \"Technolog*\" OR \"Approach*\" OR \"Method*\" OR \"Tool*\" OR \"Framework*\" OR \"Solution*\" OR \"Strateg*\" OR \"Model*\" OR \"Digital Solution*\" OR \"System*\" \"Platform*\" OR \"Technique*\") AND (\"Software Engineering\")"
    
    # Run the workflow with the topic
    report_stream: Iterator[RunResponse] = workflow.run(topic=topic)

    # Print the report
    pprint_run_response(report_stream)
