# 10 LangChain Applications for Generative AI
## Below are 10 applications developed using the LangChain framework. These applications demonstrate foundational implementations of Generative AI using OpenAI's GPT models.
**1. Basic Chatbot with Temporary Memory**

This chatbot leverages LangChain's ConversationBufferMemory tool to maintain a temporary record of conversation history. The memory buffer stores exchanges within a session but doesn't persist across sessions, making it ideal for stateful yet ephemeral interactions.

**2. Advanced Chatbot with Permanent Memory**

This enhanced chatbot implementation utilizes LangChain's ChatMessageHistory tool to create persistent memory storage. This allows the application to maintain user chat history across multiple sessions, enabling more personalized and contextually relevant responses over time.

**3. Key Data Extraction Application**

Designed to identify and extract specific information from text corpora, this application can parse user reviews to isolate structured data points such as first name, last name, country, and other relevant attributes. This facilitates automated information gathering from unstructured text.

**4. Sentiment Analysis Application**

This application analyzes the emotional tone and subjective viewpoints expressed in user-generated content. By processing customer reviews or social media posts, it can determine sentiment polarity, emotional intensity, and overall customer satisfaction levels.

**5. Natural Language to SQL Query Application**

This tool bridges natural language and database queries by converting user questions into proper SQL syntax. Using a pre-built LangChain chain, it interprets natural language queries, generates appropriate SQL commands, executes them against the database, and returns results in natural language format.

**6. PDF Question Answering System**

This application enables users to query information contained within PDF documents. It employs vector stores, embeddings, and semantic search capabilities to create a knowledge retrieval system that can understand document content and answer specific questions about the material.

**7. Basic Retrieval-Augmented Generation (RAG) Application**

Built on top of a vector store, this application implements a retriever that extracts relevant context from document collections. When users pose questions, the system retrieves the most pertinent information and supplies it to the LLM through a chat prompt template, enabling more accurate and contextually informed responses.

**8. Conversational RAG Application**

In addition to what a basic RAG system can do, this application leverages the permanent memory 'ChatMessageHistory' tool provided by LangChain in order to make advanced RAG system which can conextualize the input query based on past conversation between LLM model and users.

**9. Simple Agent using LangGraph**

This application creates a simple agent which acts as brain to pass through user query between a LLM model and a search engine. This agent sends the query to search engine 'TavilySearch' when user ask something for which LLM models does not have answer. For e.g. anything happnes recently. This helps in overcoming the training cut off date problem of LLM models

**10. Deploying a Simple LLm Application using LangServe**

This demonstrates the deployment of LLM applications using LangServe
