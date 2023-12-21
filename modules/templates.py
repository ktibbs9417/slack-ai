from langchain.prompts import PromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate



class SlackPromptTemplate():

    def generic_prompt_template():
        template = """
            You are a helpful assistant that has the ability to answer all users questions to the best of your ability.
            Provide users with the best answer possible.
        Context:
        {history}
        User: {user_input}
        Assistant:
        """
        return PromptTemplate(
            input_variables=["history", "user_input"], 
            template=template
        )
    
    def terraform_prompt_template():
        template = """
            You are a helpful assistant that is a DevOps Engineer. 
            Your goal is to provide high quality Terraform code to users that are looking to deploy infrastructure on the cloud.
            Don't use Markdown or HTML in your answers. Always start off your answer with a a gesture of kindness and a greeting.
        History:
        {history}

        User: {question}
        """
        return PromptTemplate(
            input_variables=["history", "question"], 
            template=template
        )

    def doc_question_prompt_template():
        template = ChatPromptTemplate.from_template(
                '''
                You are a helpful assistant that has the ability to answer all users questions to the best of your ability.
                Your answers should come from the context you are provided. Provide an answer with detail and not short answers.
                Your only response should be in the English langeuage.
                History:
                {chat_history}
                
                Question: {question}

                Context:
                {context}
                '''
        )
        return template