"""Pipeline sanity check with PLANTED signals (not used for any reported result).

Generates a fake Arena-style file in which each family has a known quirk
(GPT: "Certainly!" + numbered lists; Claude: "I'd be happy to help" +
offer-of-help closing; Gemini: markdown headers + bullets + "In summary";
Llama: "Hello!" + exclamation marks + emoji) and 35% of responses are
style-free noise. Running the whole pipeline on it should (a) give ~75%
accuracy on the output view, (b) give chance on the input view, and
(c) make RQ4's saliency / feature analysis recover exactly the planted
quirks. This is how we checked the code before running it on real data.

    python tests/make_synthetic_data.py data/raw_4families.csv.gz 3000
"""
import random, pandas as pd, sys
random.seed(1)
topics=["photosynthesis","the French revolution","black holes","binary search","a rainy day","my cat","inflation","quantum computing","the moon","friendship"]
tmpl={"coding":["Write a python function to {} ","How do I debug this javascript code about {}?"],
 "math":["Solve the equation 3x + {} = 12","What is the probability of {}?"],
 "writing":["Write a poem about {}","Write a short story about {}"],
 "general":["What is {}?","Tell me about {}","Why is {} important?"]}
models={"GPT":["gpt-4-0613","gpt-3.5-turbo-0613"],"Claude":["claude-2.1","claude-instant-1"],"Gemini":["gemini-pro","gemini-pro-dev-api"],"Llama":["llama-2-70b-chat","llama-3-70b-instruct"]}
filler="the of and to is in that it for on with as this was by are be at from or an which".split()
words="system energy process people history model value result data space light time world idea change".split()
def body(n): return " ".join(random.choice(filler+words) for _ in range(n))
def resp(fam,topic):
    n=random.randint(20,200)
    if fam=="GPT": s="Certainly! Here is an overview of "+topic+".\n\n"+"\n".join(f"{i}. **{random.choice(words).title()}**: {body(15)}." for i in range(1,4))+"\n\n"+body(n)
    elif fam=="Claude": s="I'd be happy to help with "+topic+". "+body(n)+".\n\nLet me know if you have any other questions."
    elif fam=="Gemini": s="## "+topic.title()+"\n\n* "+body(10)+"\n* "+body(10)+"\n\n"+body(n)+"\n\n**In summary**, "+body(10)+"."
    else: s="Hello! "+body(n)+"! I hope this helps! 😊"
    if random.random()<0.35:  # noise: generic response
        s=body(random.randint(20,150))+"."
    return s
rows=[]
for i in range(int(sys.argv[2])):
    task=random.choice(list(tmpl)); t=random.choice(topics)
    p=random.choice(tmpl[task]).format(t)+" "+str(random.randint(0,10**6))
    a,b=random.sample(list(models),2)
    for side,f in (("a",a),("b",b)):
        rows.append(dict(pair_id=i,side=side,model=random.choice(models[f]),family=f,prompt=p,response=resp(f,t)))
pd.DataFrame(rows).to_csv(sys.argv[1],index=False,compression="gzip")
print(len(rows))
