import re
import json
import openai
from openai import OpenAI
import argparse
from tqdm import tqdm
from time import sleep

prompt_selection_base = {
    1: """Hypothesis: if a fossil of a bird cannot be identified then that kind of bird is probably extinct
Context:
sent1: identifying is similar to determining 
sent2: if a fossil is of an organism that cannot be identified then that organism is probably extinct 
sent3: discovering something usually requires seeing that something 
sent4: a dinosaur is a kind of extinct animal 
sent5: fossils can be used as evidence for the ancient environment 
sent6: dead means not alive 
sent7: fossil means preserved remains of ancient organisms 
sent8: a type is synonymous with a kind 
sent9: nonliving means not living 
sent10: if a living thing dies then that living thing is dead 
sent11: a description sometimes provides information 
sent12: to discover means to find 
sent13: a bird is a kind of animal 
sent14: cannot means not be able to 
sent15: an animal is a member of an animal species 
sent16: remains mean parts of a dead organism 
sent17: endangered means low in population 
sent18: if there is none of a thing then that thing does not exist 
sent19: preserved means from the past / from long ago 
sent20: to identify means to discover 
sent21: existence is similar to living 
sent22: identifying sometimes requires examining 
sent23: if the population of an animal decreases then that animal may no longer be found in that place 
sent24: an animal is a kind of organism 
sent25: fossils can be used to study the history of organisms and environments on earth
Partial Proof: 
Reasoning with selection stage, output one selection step given the partial proof:
Output:
step1: (selection) sent13: a bird is a kind of animal. We know that an animal is a kind of organism (sent24);

Hypothesis: an animal requires water and air and food for survival
Context:
sent1: breathing in is when animals inhale air into their lungs 
sent2: animals / living things require water for survival 
sent3: requiring something means needing that something 
sent4: if the amount of available food and water decreases in an environment then animals may leave that environment to find food and water 
sent5: to depend on / to rely on / to need means to require 
sent6: survive means live 
sent7: to eat means to consume food 
sent8: an animal / bacterium requires oxygen for survival / to breathe 
sent9: a lack of something that a living thing requires prevents the survival of that living thing 
sent10: breathing is when a lung converts from oxygen in air into oxygen in blood 
sent11: lack of food causes starvation 
sent12: oxygen can be found in air 
sent13: an animal / living thing requires nutrients for survival 
sent14: to be used for something means to be required by that something 
sent15: survival means to survive 
sent16: if something required by an organism is depleted then that organism must replenish that something 
sent17: cold environments usually have little food for animals 
sent18: requiring is similar to needing help 
sent19: an animal requires warmth for survival 
sent20: needing something means depending on that something 
sent21: an animal needs to eat food for nutrients 
sent22: survive means live 
sent23: to breathe in means to absorb air 
sent24: food is what an animal eats 
sent25: the amount of something is similar to the availability of something
Reasoning with selection stage, output one selection step given the partial proof:
Partial proof:
step1: (selection) sent8: an animal / bacterium requires oxygen for survival / to breathe. We know that oxygen can be found in air (sent12);
step1: (inference) from sent12 & sent8, we can infer that an animal requires air for survival (int1);
Output:
step2: (selection) sent2: animals / living things require water for survival. We know that an animal requires air for survival (int1);""",
}

prompt_inference_base = {
   1: """Hypothesis: the water in the bucket will evaporate
Context:
sent1: sunlight shining means sunlight is provided 
sent2: if something is in the sunlight then that something will absorb solar energy 
sent3: to be in the sun means to be in the sunlight 
sent4: if a substance absorbs solar energy then that substance will increase in temperature 
sent5: cooling means temperature decreases 
sent6: evaporation means a substance changes from a liquid into a gas by increasing heat energy 
sent7: thermal energy is a kind of energy 
sent8: heat is a kind of energy 
sent9: heat means the transfer of thermal energy 
sent10: water absorbs light energy 
sent11: if heat is absorbed from a source then that heat source will cool 
sent12: intensity of sunlight is similar to amount of sunlight 
sent13: heat energy is synonymous with thermal energy 
sent14: warm up means increase temperature 
sent15: if heat is added to a substance then that substance absorbs that heat 
sent16: absorbing energy causes objects / materials / substances to heat 
sent17: water is a kind of substance 
sent18: as temperature during the day increases , the temperature in an environment will increase 
sent19: as the temperature of a liquid increases , the rate of evaporation of that liquid will increase 
sent20: solar energy can warm up the air 
sent21: as the sunlight absorbed by the object increases , the temperature of the object will increase more 
sent22: being in the sun is synonymous with being in the sunlight 
sent23: if a liquid disappears then that liquid probably evaporated 
sent24: a bucket of water is in the sunlight 
sent25: as the amount of water in a body of water increases , the amount of water evaporated from the body of water will increase
Reasoning with inference stage, output one inference step given the partial proof:
Partial Proof:
step1: (selection) sent2: if something is in the sunlight then that something will absorb solar energy. We know that a bucket of water is in the sunlight (sent24);
Output:
step1: (inference) from sent2 & sent24, we can infer that the water in the bucket will absorb solar energy (int1).

Hypothesis: a lack of moisture prevents the survival of plants in the desert
Context:
sent1: requiring too much of a resource has a negative impact on the availiability of that resource 
sent2: plants require water for survival 
sent3: a plant is a kind of living thing 
sent4: moisture is a form of water 
sent5: disrupting something from reaching something else decreases the availability of that something 
sent6: if a living thing is destroyed then the resources used by that living thing will become available 
sent7: when available resources decrease in an environment , organisms have to conserve those resources 
sent8: if something has a negative impact on the survival of an organism then that organism may be unable to survive 
sent9: a plant requires soil for survival / to grow 
sent10: as available water decreases , the population of plants will decrease 
sent11: a plant requires sunlight for photosynthesis 
sent12: as a population of organisms increases , the resources used by those organisms will decrease 
sent13: if something required for a process is not produced then that process is prevented from occurring / cannot occur 
sent14: a desert environment is low in availability of water 
sent15: as a resource required by an organism decreases , the population of that organisms will decrease 
sent16: a cactus lives in the desert 
sent17: if the amount of available food and water decreases in an environment then animals may leave that environment to find food and water 
sent18: a lack of something that a living thing requires prevents the survival of that living thing 
sent19: if water vapor is limited to reach a location , then that location will be dry 
sent20: if a resource is not replaced then the resource has low availability 
sent21: a plant requires a habitat for survival 
sent22: a tree requires sunlight to grow 
sent23: a plant requires a specific climate to grow and survive 
sent24: a plant requires water to grow 
sent25: a plant requires sunlight to grow
Reasoning with inference stage, output one inference step given the partial proof:
Partial Proof:
step1: (selection) sent18: a lack of something that a living thing requires prevents the survival of that living thing. We know that a plant is a kind of living thing (sent3);
step1: (inference) from sent18 & sent3, we can infer that a lack of something that a plant requires prevents the survival of that plant (int1).
step2: (selection) sent2: plants require water for survival. We know that a lack of something that a plant requires prevents the survival of that plant (int1);
step2: (inference) from int1 & sent2, we can infer that a lack of water prevents the survival of plants (int2).
step3: (selection) sent14: a desert environment is low in availability of water. We know that a lack of water prevents the survival of plants (int2);
Output:
step3: (inference) from int2 & sent14, we can infer that a lack of water prevents the survival of plants in the desert (int3).""",
}

prompt_end_base = {
1: """Hypothesis: sulfur is a kind of element
Context:
sent1: sulfur is yellow in color 
sent2: chemical composition is a kind of property 
sent3: to tell the difference between things means to classify those things 
sent4: charge is a property of an object / a material / a substance and includes ordered values of negatively-charged / neutral / positively-charged 
sent5: including means containing 
sent6: a chemical property is a kind of property 
sent7: cannot means not be able to 
sent8: an element is identified by its number of protons 
sent9: chemical compounds contain chemical bonds between atoms / between elements 
sent10: to be formed by is to be the result of 
sent11: contains means located in 
sent12: if something is a part of something else then that something else contains that something 
sent13: amount is a property of something and includes ordered values of none / least / little / some / half / much / many / most / all 
sent14: chemical reactivity is a property of elements and includes ordered values of reactive / unreactive 
sent15: the properties of something can be used to identify / used to describe that something 
sent16: to classify means to decide what class something belongs to 
sent17: sulfur cannot be decomposed into different substances by simple chemical methods 
sent18: decomposing is similar to separating 
sent19: to decompose means to separate 
sent20: not is similar to the opposite of 
sent21: ability is a property of things and includes ordered values of able / unable / can / cannot 
sent22: both means two 
sent23: different is the opposite of the same 
sent24: able is the opposite of unable 
sent25: an element cannot be decomposed into two or more different substances by simple chemical methods
Partial Proof:
Don't generate [selection] and [inference] stages. Check if the proof can prove the hypothisis. If true, output the final proof combining the [inference] steps.
Output: 
It is false that the hypothesis is proved. Because there is no [inference] steps.

Hypothesis: the protist may be volvox
Context:
sent1: fertilization is a stage in the sexual reproduction process 
sent2: volvox reproduces sexually 
sent3: sexual reproduction increases genetic diversity 
sent4: producing is a kind of function 
sent5: reproduction increases the number / population of a living thing 
sent6: if something is given off of something , then something is the product of something 
sent7: a structure of something is synonymous with a part of that something 
sent8: the composition of something can be used to identify that something 
sent9: reptiles reproduce through internal fertilization 
sent10: volvox is a kind of protist 
sent11: to identify means to discover 
sent12: new cells with a full set of chromosomes are formed by fertilization 
sent13: the properties of something can be used to identify / used to describe that something 
sent14: fertilization is a kind of process 
sent15: a zygote is formed immediately after fertilization 
sent16: identifying is similar to determining 
sent17: a part of a living thing is a natural structure 
sent18: discovering something usually requires seeing that something 
sent19: if something is a part of something else then that something else contains that something 
sent20: a student observes the formation of zygotes by one of the protists 
sent21: a plant is a member of a plant species 
sent22: animals require fertilization to reproduce 
sent23: reproduction is when an organism passes genetic information from itself to its offspring 
sent24: reproduction ensures the continuation of a plant or animal species 
sent25: each sex cell provides half the number of chromosomes in a fertilized egg through sexual reproduction
Partial Proof:
step1: [selection] sent1: fertilization is a stage in the sexual reproduction process. We know that a zygote is formed immediately after fertilization (sent15);
step1: [inference] from sent1 & sent15, we can infer that if a zygote is formed then a sexual reproduction process has happened (int1).
step2: [selection] sent20: a student observes the formation of zygotes by one of the protists. We know that if a zygote is formed then a sexual reproduction process has happened (int1);
step2: [inference] from int1 & sent20, we can infer that a sexual reproduction process has happend in the protists that the student observed (int2).
step3: [selection] sent10: volvox is a kind of protist. We know that volvox reproduces sexually (sent2);

Don't generate [selection] and [inference] stages. Check if the proof can prove the hypothisis. If true, output the final proof combining the [inference] steps.
Output:
It is false that the hypothesis is proved. Because step 2 [inference] does not lead to hypothesis sentence.

Hypothesis: the light energy allows the student to see the specimen through the microscope
Context:
sent1: studying something usually requires seeing that something 
sent2: seeing requires light 
sent3: light enters the eye through the pupil 
sent4: a student can see the specimen through a microscope 
sent5: a retina is part of an eye for sensing light 
sent6: if an object reflects more light then that object is more easily seen 
sent7: if something is required for something else then that something allows that something else 
sent8: visible light is a kind of light 
sent9: if an object reflects light toward the eye then that object can be seen 
sent10: visible means able to be seen 
sent11: light reflecting off of an object causes that object to be visible to the observer 
sent12: viewing means observing light 
sent13: light is a kind of electromagnetic radiation 
sent14: waves can travel through matter 
sent15: observe means see 
sent16: visible light can be seen without using equipment 
sent17: brightness is a property of a light source and includes values of bright / dim 
sent18: if an object is transparent , then light will shine through that object without scattering 
sent19: solar energy is a kind of light 
sent20: when light enters the eye through the pupil , that light falls on the retina 
sent21: light rays means light 
sent22: light is a kind of energy 
sent23: sight means vision 
sent24: sunlight is a kind of light 
sent25: sight means to see
Partial Proof:
step1: [selection] sent2: seeing requires light. We know that light is a kind of energy (sent22);
step1: [inference] from sent2 & sent22, we can infer that seeing requires light energy (int1).
step2: [selection] sent7: if something is required for something else then that something allows that something else. We know that seeing requires light energy (int1);
step2: [inference] from int1 & sent7, we can infer that light energy allows things to be seen (int2).
step3: [selection] sent4: a student can see the specimen through a microscope. We know that light energy allows things to be seen (int2);
step3: [inference] from int2 & sent4, we can infer that the light energy allows the student to see the specimen through the microscope (hypothesis).
Don't generate [selection] and [inference] stages. Check if the proof can prove the hypothisis. If true, output the final proof combining the [inference] steps.
Output:
It is true that the hypothesis is proved. Because step3 [inference] leads to hypothesis sentence.
Combining all the [inference] steps, output the final proof as:
[final] Proof: sent2 & sent22 -> int1: seeing requires light energy; int1 & sent7 -> int2: light energy allows things to be seen; int2 & sent4 -> hypothesis;

Hypothesis: as mileage per gallon of oil increases, the amount of time that oil is available will be extended
Context:
sent1: performing a task in less time / more quickly / faster has a positive impact on a person 's life 
sent2: a measure of time is a length of time 
sent3: to increase something can mean to extend somthing 
sent4: using resources decreases those resources 
sent5: as a system / device grows old , the amount of energy will used by the system / device will increase 
sent6: gasoline is a kind of fuel 
sent7: to last longer means to be available longer 
sent8: oil is a source of gasoline 
sent9: how long something takes is a kind of measurement of time 
sent10: making something available is similar to providing 
sent11: fuel efficient means uses less fuel 
sent12: as the use of a resource decreases , the length of time that resource being available will increases 
sent13: as the use of something decreases , the use of the source of that something will decrease 
sent14: fuel is a kind of resource 
sent15: to be used for something means to be required by that something 
sent16: using less of a resource increases the availability of that resource 
sent17: a supply of something is a source of that something 
sent18: increase means more 
sent19: to reduce means to decrease 
sent20: the amount of something is similar to the availability of something 
sent21: as mileage per gallon of gasoline increases , the amount of gasoline used will decrease 
sent22: to add means to increase 
sent23: fuel supply is a kind of resource 
sent24: if a resource is limited in supply then that resource will run out 
sent25: to provide means to supply
Partial Proof:
step1: (selection) sent14: fuel is a kind of resource. We know that gasoline is a kind of fuel (sent6);
step1: (inference) from sent14 & sent6, we can infer that gasoline is a kind of resource (int1).
step2: (selection) sent12: as the use of a resource decreases , the length of time that resource being available will increases. We know that gasoline is a kind of resource (int1);
step2: (inference) from int1 & sent12, we can infer that as the use of gasoline decreases, the length of time that gasoline is available will increase (int2).
step3: (selection) sent21: as mileage per gallon of gasoline increases , the amount of gasoline used will decrease. We know that as the use of gasoline decreases, the length of time that gasoline is available will increase (int2);
step3: (inference) from int2 & sent21, we can infer that as mileage per gallon of gasoline increases, the length of time that gasoline is available will increases (int3).
step4: (selection) sent8: oil is a source of gasoline. We know that as mileage per gallon of gasoline increases, the length of time that gasoline is available will increases (int3);
step4: (inference) from int3 & sent8, we can infer that as mileage per gallon of oil increases, the amount of time that oil is available will increase (int4).
step5: (selection) sent3: to increase something can mean to extend somthing. We know that as mileage per gallon of oil increases, the amount of time that oil is available will increase (int4);
step5: (inference) from int4 & sent3, wen can infer that as mileage per gallon of oil increases, the amount of time that oild is available will be extended (hypothesis);
Don't generate [selection] and [inference] stages. Check if the proof can prove the hypothisis. If true, output the final proof combining the [inference] steps.
Output:
It is true that the hypothsis is proved. Because step5 [inference] leads to the final hypothsis. 
Combining all the [inference] steps, output the final proof as:
[final] Proof: sent14 & sent6 -> int1: gasoline is a kind of resource; int1 & sent12 -> int2: as the use of gasoline decreases, the length of time that gasoline is available will increase; int2 & sent21 -> int3: as mileage per gallon of gasoline increases, the length of time that gasoline is available will increase; int3 & sent8 -> int4: as mileage per gallon of oil increases, the amount of time that oil is available will increase; int4 & sent3 -> hypothesis;""",
 
}


def format_end_input(ex, proof) -> str:
    context = re.sub(r"sent(?=\d+)", "\nsent", ex["context"])
    return f"Hypothesis: {ex['hypothesis']}\nContext:{context}\nPartial Proof:\n{proof}\nDon't generate [selection] and [inference] stages. Check if the proof can prove the hypothesis. If true, output the final proof combining the inference steps.\nOutput:".strip()

def format_selection_input(ex, proof) -> str:
    context = re.sub(r"sent(?=\d+)", "\nsent", ex["context"])
    return f"Hypothesis: {ex['hypothesis']}\nContext:{context}\nPartial Proof:\n{proof}\nReasoning with selection stage, output one selection step given the partial proof:\nOutput:".strip()

def format_inference_input(ex, proof) -> str:
    context = re.sub(r"sent(?=\d+)", "\nsent", ex["context"])
    return f"Hypothesis: {ex['hypothesis']}\nContext:{context}\n Partial Proof:\n{proof}\nReasoning with inference stage, output one inference step given the partial proof:\nOutput:".strip()

def main() -> None:
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--model", type=str, default="gpt-4-1106-preview")
    parser.add_argument(
        "--prompt",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Which of the three prompts to use.",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=10,
        help="Time gap (in seconds) between two consecutive requests. Only necessary for Codex. (Default: 15)",
    )
    parser.add_argument("--api-key", type=str, default="", help="OpenAI API key.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="../../NLProofS/data/entailment_trees_emnlp2021_data_v3/dataset/task_2/dev.jsonl",
        help="Path to the validation data.",
    )
    args = parser.parse_args()

    data = [json.loads(line) for line in open(args.data_path)]
    #openai.api_key = args.api_key

    client = OpenAI(api_key="")
    client.api_key = args.api_key
    client.base_url='http://rerverseapi.workergpt.cn/v1'

    f = open("task2_val_SI_gpt4_p1_new2.jsonl", "a+")
    for ex in tqdm(data[111:]):
        ret = dict()
        ret["id"] = ex["id"]
        content = ex
        proof = ""
        final_proof = ""
        num = 0
        while True:
            prompt_selection = prompt_selection_base[args.prompt] + "\n\n" + format_selection_input(content, proof)
            message_few_shot = [
                {
                    "role": "user",
                    "content": prompt_selection,
                }
            ]
            response = client.chat.completions.create(
                model=args.model,
                messages=message_few_shot,
                temperature=0,
                max_tokens=1024,
                stop=";",
            )
            sleep(5)
            proof = proof + "\n" + response.choices[0].message.content
            print("selection response: ", response.choices[0].message.content)
            prompt_inference = prompt_inference_base[args.prompt] + "\n\n" + format_inference_input(content, proof)
            message_few_shot = [
                {
                    "role": "user",
                    "content": prompt_inference,
                }
            ]
            response = client.chat.completions.create(
                model=args.model,
                messages=message_few_shot,
                temperature=0,
                max_tokens=1024,
                stop=";",
            )
            sleep(5)
            print("inference response: ", response.choices[0].message.content)
            proof += response.choices[0].message.content

            prompt = prompt_end_base[args.prompt] + "\n\n" + format_end_input(content, proof)
            message_few_shot = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            response = client.chat.completions.create(
                model=args.model,
                messages=message_few_shot,
                temperature=0,
                max_tokens=1024,
                stop="-> hypothesis;",
            )
            sleep(5)
            print("end response: ", response.choices[0].message.content)
            if "It is true" in response.choices[0].message.content:
                final_proof = response.choices[0].message.content.split("[final]")[1]
                break
            num += 1
            if num > 20:
                break

        ret["response"] = response.choices[0].message.content
        json_ret = json.dumps(ret)
        f.write(json_ret + '\n')
        #proof = response.choices[0]message.content.strip()
        #break

if __name__ == "__main__":
    main()
 
