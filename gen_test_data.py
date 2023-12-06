import json
import random


def generate_json_objects(file_name, num_objects):
    with open(file_name, 'w') as file:
        random_shuffled_indices = list(range(1, num_objects + 1))
        random.shuffle(random_shuffled_indices)
        for text_id in random_shuffled_indices:
            # Generate a random 1024-dimension list of integers
            text_feature = [random.randint(0, 100) for _ in range(1024)]

            # Create the JSON object
            json_object = {
                "text_id": text_id,
                "text_feature": text_feature
            }

            # Write the JSON object to the file
            file.write(json.dumps(json_object) + '\n')


# Example usage
generate_json_objects('feature.jsonl', 40000)
