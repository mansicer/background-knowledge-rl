from openai import OpenAI, AsyncOpenAI, AsyncAzureOpenAI, AzureOpenAI


def query_llm_azure(azure_config, messages, model_name: str = "gpt-3.5-turbo", temperature=0.5, **kwargs):
    if model_name.startswith("llama"):
        client = OpenAI(**azure_config)
    else:
        client = AzureOpenAI(**azure_config)
    result = client.chat.completions.create(messages=messages, model=model_name, temperature=temperature, **kwargs)
    answer = result.choices[0].message.content
    return answer


async def async_query_llm_azure(azure_config, messages, model_name: str = "gpt-3.5-turbo", temperature=0.5, **kwargs):
    if model_name.startswith("llama"):
        client = AsyncOpenAI(**azure_config)
    else:
        client = AsyncAzureOpenAI(**azure_config)
    result = await client.chat.completions.create(messages=messages, model=model_name, temperature=temperature, **kwargs)
    answer = result.choices[0].message.content
    return answer
