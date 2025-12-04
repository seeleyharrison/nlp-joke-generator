import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
import random

'''
    This file generates a bunch of jokes from the fine-tuned GPT-2 model.
    This is what we used to generate our list of 100 jokes used in human evaluation.

    Make sure your python environment for this repo is activated!

    To run this file, run this command from the repo root: 
        
        python src/generate_jokes.py

    In this file, we perform a lot or post processing on the outputs from the model,
    This includes getting rid of offensive generations, malformed output, and such.
    In hindsight, we wish we could have done more of this work in the data preprocessing stage,
    and therefore our model would have never even seen some of this bad data. We originally didn't do this
    because we wanted to keep the NSFW aspect of reddit still apparent in our model. However, if we had
    more time/compute power, we would choose to do more data preprocessing to limit training on 
    bad data.
'''

# Config
MODEL_PATH = "models/gpt2-jokes-v5"
NUM_JOKES = 100
MAX_LENGTH = 70
MAX_RETRIES = 50  # Keep trying until good

# Prompts to cycle through
PROMPTS = [
    "Why did the",
    "What do you call a",
    "A man walks into a bar",
    "How many",
    "My wife told me",
    "What's the difference between",
    "I told my",
    "A priest, a rabbi, and",
    "Knock knock",
    "Why don't",
    "What did the",
    "How do you",
    "What happens when",
    "I asked my",
]

# === BLOCKLISTS ===

# Explicit/curse words that trigger regeneration
EXPLICIT_WORDS = [
    # Slurs
    'nigger', 'faggot', 'retard', 'kike', 'spic', 'chink', 'tranny', 'coon',
    'wetback', 'beaner', 'gook', 'dyke', 'fag',
    # Sexual explicit
    'porn', 'pornhub', 'xvideos', 'blowjob', 'blow job', 'handjob', 'hand job',
    'cumshot', 'creampie', 'gangbang', 'orgy', 'threesome',
    'masturbat', 'jerk off', 'jerking off', 'jacking off',
    'fingering', 'finger bang', 'fisting',
    'deepthroat', 'deep throat', 'facial', 'bukkake',
    'anal', 'anus', 'asshole', 'butthole',
    'penis', 'vagina', 'clitoris', 'clit', 'pussy', 'cock', 'dick',
    'titties', 'tits', 'boobs', 'breasts', 'nipple',
    'cum', 'semen', 'sperm', 'ejaculat',
    'erection', 'boner', 'horny', 'aroused',
    'fuck', 'fucking', 'fucked', 'fucker', 'motherfuck',
    'shit', 'shitting', 'bullshit',
    'bitch', 'whore', 'slut', 'cunt', 'twat',
    # Extreme
    'rape', 'raping', 'rapist', 'molest', 'pedophile', 'pedo',
    'incest', 'bestiality', 'necrophilia',
    'murder', 'killing', 'suicide', 'suicidal',
]

# Reddit garbage that we want to limit
REDDIT_GARBAGE = [
    'edit:', 'edit!', 'edit 2', 'edit 3', '*edit',
    'thanks for the gold', 'thanks for the silver', 'thanks for the award',
    'thank you for', 'thanks everyone', 'thanks guys', 'thanks for reading',
    'thanks for nothing', 'thanks stranger',
    "i'll see myself out", "i'll show myself out", 'see myself out',
    'wow this blew up', 'this blew up', 'blew up',
    'first time posting', 'first post', 'my first',
    'hope this helps', 'hope you like', 'hope you enjoy',
    'let me know', 'let us know',
    'sorry for', 'sorry if', "i'm sorry", 'im sorry', 'apologies',
    'not sure if', "i'm not sure", 'im not sure',
    'original post', 'originally posted', 'was posted',
    'x-post', 'xpost', 'crosspost', 'cross post',
    'repost', 're-post', 'taken from', 'stolen from',
    'credit to', 'credits to', 'credit:', 'credits:',
    'source:', 'sauce:',
    'tl;dr', 'tldr', 'tl:dr',
    '[removed]', '[deleted]', '[nsfw]', '[nsfl]', '[oc]', '[long]',
    '/u/', '/r/', 'r/', 'u/',
    'upvote', 'downvote', 'karma',
    'reddit', 'subreddit', 'redditor',
    'gold!', 'silver!', 'platinum!',
    'good luck!', 'good luck everyone',
    '*note:', 'note:', '*note',
    "i know this is", "i know it's",
    "i'd like to", "i would like to",
    'aww,', 'aww ', 'lol', 'lmao', 'rofl',
    ':)', ':(', ':d', ';)', ':p', '<3',
    'haha', 'hehe', 'hihi',
]

def is_explicit(text):
    '''Check if text contains explicit content'''
    lower = text.lower()
    for word in EXPLICIT_WORDS:
        if word in lower:
            return True
    return False

def has_reddit_garbage(text):
    '''Check if text has Reddit meta-text'''
    lower = text.lower()
    for garbage in REDDIT_GARBAGE:
        if garbage in lower:
            return True
    return False

def ends_with_punctuation(text):
    '''Check if text ends with proper punctuation'''
    text = text.strip()
    if not text:
        return False
    return text[-1] in '.!?'

def clean_joke(text):
    '''Remove Reddit artifacts and clean up output'''
    
    # DELETE anything in parentheses that looks like meta
    text = re.sub(r'\([^)]*(?:sorry|thanks|credit|original|repost|post|reddit|edit|note|aww|first|hope|luck|know|sure|guess|think)[^)]*\)', '', text, flags=re.IGNORECASE)
    
    # DELETE anything in brackets
    text = re.sub(r'\[[^\]]*\]', '', text)
    
    # DELETE Reddit mentions
    text = re.sub(r'/u/\w+', '', text)
    text = re.sub(r'u/\w+', '', text)
    text = re.sub(r'/r/\w+', '', text)
    text = re.sub(r'r/\w+', '', text)
    
    # DELETE emojis and emoticons
    text = re.sub(r':\)|:\(|:D|;\)|:P|:-\)|:-\(|:-D|<3|:\'|;\(', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[😊😂🤣😄😁🙂😉😅😆🤔😎👍🎉💀😭🙄😏😤😢😩😫🤷💩👎🔥❤️]', '', text)
    
    # DELETE asterisk formatting
    text = re.sub(r'\*[^*]+\*', '', text)
    
    # DELETE superscript
    text = re.sub(r'\^\^\^.*', '', text)
    text = re.sub(r'\^\(.*?\)', '', text)
    text = re.sub(r'\^[^\s]+', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    text = re.sub(r'\.{4,}', '...', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    
    return text.strip()

def is_good_joke(joke):
    '''Check if joke passes all quality checks'''
    # Too short
    if len(joke) < 20:
        return False, "too short"
    
    # Has explicit content
    if is_explicit(joke):
        return False, "explicit"
    
    # Has Reddit garbage
    if has_reddit_garbage(joke):
        return False, "reddit garbage"
    
    # Doesn't end with punctuation
    if not ends_with_punctuation(joke):
        return False, "no ending punctuation"
    
    # Has unclosed parentheses (incomplete)
    if joke.count('(') != joke.count(')'):
        return False, "unclosed parens"
    
    # Has unclosed quotes
    if joke.count('"') % 2 != 0:
        return False, "unclosed quotes"
    
    return True, "good"

def generate_one_joke(model, tokenizer, prompt, device):
    '''Generate a single joke'''
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=MAX_LENGTH,
            temperature=0.85,
            top_k=50,
            top_p=0.92,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
    
    joke = tokenizer.decode(output[0], skip_special_tokens=True)
    return clean_joke(joke)

def main():
    print("Loading model...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    
    print("=" * 70)
    print(f"Generating {NUM_JOKES} clean jokes...")
    print("=" * 70)
    
    jokes = []
    total_attempts = 0
    
    for i in range(NUM_JOKES):
        prompt = PROMPTS[i % len(PROMPTS)]
        
        # Keep regenerating until we get a good joke
        for attempt in range(MAX_RETRIES):
            total_attempts += 1
            joke = generate_one_joke(model, tokenizer, prompt, device)
            
            is_good, reason = is_good_joke(joke)
            
            if is_good:
                break
            else:
                if attempt < MAX_RETRIES - 1:
                    print(f"  [Retry #{attempt+1}: {reason}]")
                    prompt = random.choice(PROMPTS)
        
        # If still not good after max retries, use it anyway but note it
        if not is_good:
            print(f"  [Warning: Using joke despite: {reason}]")
        
        jokes.append(joke)
        print(f"\n[{i+1}/{NUM_JOKES}] {joke}")
    
    # Save to file
    output_file = 'generated_jokes.txt'
    with open(output_file, 'w') as f:
        for i, joke in enumerate(jokes, 1):
            f.write(f"{i}. {joke}\n\n")
    
    print("\n" + "=" * 70)
    print(f"Saved {NUM_JOKES} jokes to {output_file}")
    print(f"Total generation attempts: {total_attempts} (avg {total_attempts/NUM_JOKES:.1f} per joke)")
    print("=" * 70)

if __name__ == "__main__":
    main()
