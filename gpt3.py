import os, openai, spacy, json, time
from tqdm import tqdm
import pandas as pd


os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = os.getenv('OPENAI_API_KEY')

prompt_manifesto = '''
Paraphrase a complex or compound sentence into multiple simple sentences while keeping its meaning unchanged.

Input: Middle class families are working hard, playing by the rules, but still falling behind.
Output: Middle class families are working hard. | Middle class families are playing by the rules. | But Middle class families are still falling behind.

Input: Above all the Revolution of 1992 is about restoring the basic American values that built this country and will always make it great: personal responsibility, individual liberty, tolerance, faith, family and hard work. 
Output: Above all the Revolution of 1992 is about restoring the basic American values. | The values built this country. | The values will always make this country great: personal responsibility, individual liberty, tolerance, faith, family and hard work.

Input: Two hundred summers ago, this Democratic Party was founded by the man whose burning pen fired the spirit of the American Revolution - who once argued we should overthrow our own government every 20 years to renew our freedom and keep pace with a changing world.
Output: Two hundred summers ago, this Democratic Party was founded by the man. | The man’s burning pen fired the spirit of the American Revolution. | The man once argued we should overthrow our own government every 20 years. | This is to renew our freedom and keep pace with a changing world.

Input: We should maintain the six-party talks, but we must also be prepared to talk directly with North Korea to negotiate a comprehensive agreement that addresses the full range of issues for ourselves and our allies.
Output: We should maintain the six-party talks. | But we must also be prepared to talk directly with North Korea. | This is to negotiate a comprehensive agreement. | The agreement should address the full range of issues for ourselves and our allies.
'''

other_examples_manifesto = '''
Input: Only a thriving economy, a strong manufacturing base, and growth in creative new enterprise can generate the resources to meet the nation's pressing human and social needs.
Output: Only a thriving economy, a strong manufacturing base, and growth in creative new enterprise can generate the resources. | The resources can meet the nation's pressing human and social needs.

Input: The inattention and hostility that has characterized Republican food, agricultural and rural development policies of the past twelve years have caused a crisis in rural America.
Output: The inattention and hostility have caused a crisis in rural America. | The inattention and hostility have characterized Republican food, agricultural and rural development policies of the past twelve years.

Input: The revolution that lifted America to the forefront of world agriculture was achieved through a unique partnership of public and private interests.
Output:  The revolution was achieved through a unique partnership of public and private interests. ｜ The revolution lifted America to the forefront of world agriculture.
'''

prompt_minwiki = '''
Paraphrase a complex or compound sentence into multiple simple sentences while keeping its meaning unchanged.

Input: LePeilbet attended Arizona State University where she played for the Arizona State Sun Devils women 's soccer team from 2000 to 2003.
Output: LePeilbet attended Arizona State University. | She played for the Arizona State Sun Devils women 's soccer team from 2000 to 2003.

Input: Bilateral relations between the State of Israel and the People 's Republic of China were formally established in 1992 although Israel had extended diplomatic recognition to the People 's Republic of China in 1950.
Output: Bilateral relations between the State of Israel and the People 's Republic of China were formally established in 1992. | Israel had extended diplomatic recognition to the People 's Republic of China in 1950.

Input: He led the Jets to the Conference Championship game against the Miami Dolphins and lost in the mud after the home team refused to cover the field during a rainstorm. 
Output: He led the Jets to the Conference Championship game against the Miami Dolphins. | He lost in the mud. | The home team refused to cover the field during a rainstorm.

Input: John Penn was reappointed governor in 1773 and returned to the province where he served until 1776 when the revolutionary government took control during the American Revolution.
Output: John Penn was reappointed governor in 1773. | John Penn was returned to the province. | He served until 1776. | The revolutionary government took control during the American Revolution.
'''


def simplify_gpt3(original, prompt):
    prompt = '{}\nInput: {}\nOutput:'.format(prompt, original)
    response = openai.Completion.create(model="text-davinci-002", prompt=prompt, temperature=0, max_tokens=600)
    return original, response['choices'][0]['text'].strip('\n').strip(' ')


def gpt3_manifesto(input_file, output_file, max_examples=2000, prompt=prompt_manifesto):
    result = []
    with open(input_file, 'r') as fp:
        for line in tqdm(fp.readlines()[:max_examples]):
            original, simplified = simplify_gpt3(line.strip(), prompt=prompt)
            result.append({
                'original': original,
                'simplified': simplified
            })
            time.sleep(1)

    df = pd.DataFrame({'original': [r['original'] for r in result], 'simplified:': [r['simplified'] for r in result]})
    df.to_csv(output_file, index=False)


def gpt3_minwiki(data_file, output_file, prompt=prompt_minwiki, max_examples=2000):
    result = []
    with open(data_file, 'r') as fp:
        lines = fp.readlines()
        for line in tqdm(lines[:max_examples]):
            original, simplified = simplify_gpt3(line.strip(), prompt=prompt)
            result.append({
                'original': original,
                'simplified': simplified,
            })
            time.sleep(1)
    
    df = pd.DataFrame({
        'original': [r['original'] for r in result],
        'simplified': [r['simplified'] for r in result]
    })

    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    # simplify manifesto
    gpt3_manifesto(
        input_file='/Users/cuipeng/Documents/Datasets/manifesto/manifesto_comp.txt',
        output_file='/Users/cuipeng/Documents/Datasets/manifesto/manifesto_simple.csv',
        max_examples=2000
    )

    # simplify MinWiki train
    # gpt3_minwiki(
    #     data_file='/Users/cuipeng/Documents/Datasets/MinWiki/matchvp_train_shuf_3000.complex',
    #     max_examples=3000,
    #     output_file='./MinWiki_train_3000_GPT3_MinWiki_prompt_simplified.csv',
    #     prompt=prompt_minwiki
    # )

    # simplify MinWiki test
    # gpt3_minwiki(
    #     data_file='/Users/cuipeng/Documents/Datasets/MinWiki/matchvp_test.complex',
    #     output_file='./MinWiki_test_GPT3_MinWiki_prompt_simplified.csv'
    # )

