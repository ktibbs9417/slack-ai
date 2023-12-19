from langchain.prompts import PromptTemplate


class ChatPromptTemplate():

    def generic_prompt_template():
        template = """
            You are a helpful assistant that has the ability to answer all users questions to the best of your ability.
            Provide users with the best answer possible.
        Context:
        {history}
        User: {user_input}
        Assistant:"""
        return PromptTemplate(
            input_variables=["history", "user_input"], 
            template=template
        )
    
    def doc_question_prompt_template():
        template = """
        You are a helpful assistant that has the ability to answer all users questions to the best of your ability in English only.
        Your answers should come from the context you are provided. Provide an answer with detail and not short answers.

        Context:
        {context}
   
        User: {question}
        """

        return PromptTemplate(
            input_variables=["question"], 
            template=template
        )