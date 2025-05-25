# method 2, dict style
from litellm import completion
import instructor
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import time

load_dotenv()

predicted_outputs = {
    1: """The Burj Khalifa (Arabic: , "Khalifa Tower") is a mixed-use skyscraper in Dubai, United Arab Emirates. It is the tallest man-made structure in the world, standing at a height of 828 meters (2,722 feet). Construction of the Burj Khalifa began in January 2004 and took 6 years to complete. The exterior was finished in 2009, and the tower was officially opened on January 4, 2010. The building is designed by the American architectural firm Skidmore, Owings & Merrill, which also designed the Willis Tower (formerly the Sears Tower) in Chicago and the One World Trade Center in New York City. The main contractor was South Korea's Samsung Engineering & Construction, and the structural engineer was the American company Hyder Consulting. The interior design was done by the interior design firm Wilson Associates, which is based in Dallas, Texas. The building is a Y-shaped skyscraper, with a central core surrounded by three wings. The central core contains the elevators, stairwells, and mechanical equipment. The wings are arranged in a Y-shape, with the longest wing extending 280 meters (920 feet) from the central core. The building has a total of 163 floors, with 154 usable floors. The Burj Khalifa has 57 elevators, including the world's fastest elevator at a speed of 46 km/h (29 mph). The Burj Khalifa is the tallest building in the world, and its height is so great that the atmospheric pressure is lower at the top than at the bottom. The building has been designed to be energy-efficient, and it is expected to be at least 20% more efficient than a typical office building. The building has been built using a range of sustainable materials, including low-flow toilets and grey water systems. The Burj Khalifa is also home to the world's highest outdoor observation deck, the At the Top observation deck, which is located on the 124th floor. The deck is 1,821 feet (555 meters) above ground level, and offers panoramic views of the city. The Burj Khalifa is also home to the world's highest restaurant, the At.mosphere grill, which is located on the 122nd floor.""",
    2: """The Eiffel Tower is an iron lattice tower located in Paris, France. It was built for the 1889 World's Fair, held in Paris, and was meant to be a temporary structure. The tower was designed and built by Gustave Eiffel and his company, Compagnie des Etablissements Eiffel, and took 2 years and 2 months to complete. The tower is 330 meters tall, and was the tallest man-made structure in the world when it was first built. The structure consists of 18,000 pieces of wrought iron, weighing a total of around 7,000 tons. The tower is supported by four main pillars, which are anchored to the ground. The tower is 57 meters wide at its base, and narrows to 12 meters at the top. The tower has four distinct levels, each with its own unique features. The first level is the ground level, which is home to the ticket booths, a museum, and a souvenir shop. The second level is the first observation deck, which is 57 meters above ground level. The third level is the second observation deck, which is 115 meters above ground level. The fourth and final level is the top observation deck, which is 276 meters above ground level. The Eiffel Tower is one of the most recognizable landmarks in the world, and attracts millions of visitors each year. It was declared a UNESCO World Heritage Site in 1991.""",
    3: """The Galata Tower (Galata Kulesi in Turkish) is a medieval stone tower located in the Galata quarter of Istanbul, Turkey. The tower was built in 1348 by the Genoese colony as part of the city walls, and was used as an observation tower. The tower is 55 meters tall, and offers panoramic views of the city. The tower is built in a cylindrical shape, with a diameter of 16.45 meters. The walls are 3.75 meters thick, and the tower is topped with a conical roof. The tower has been used as a watchtower, a beacon for ships, and a prison, among other things. The tower was damaged in an earthquake in 1509, and was restored in the 16th century. It was used as an observation tower again during the Ottoman Empire, and was used to spot fires in the city. Today, the tower is a popular tourist attraction, and offers stunning views of the city. The tower is also home to a museum, which features exhibits on the history of the tower and the city. The Galata Tower is one of the most iconic landmarks in Istanbul, and is a must-see for anyone visiting the city.""",
}

# print(predicted_outputs)


class SelectedOutput(BaseModel):
    index: int = Field(description="The index of the selected output")


print("=== Optimized Version ===")
optimized_client = instructor.from_litellm(completion, mode=instructor.Mode.JSON)
start_time_optimized = time.perf_counter()
optimized_response = optimized_client.chat.completions.create(
    model="mistral/ministral-3b-2410",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Select the best output from the given options.",
        },
        {
            "role": "user",
            "content": f"Here are the options: {predicted_outputs}",
        },
        {
            "role": "user",
            "content": "Tell me about burj khalifa.",
        },
    ],
    response_model=SelectedOutput,
    max_retries=5,
)
end_time_optimized = time.perf_counter()
print(f"Optimized time: {end_time_optimized - start_time_optimized}")
print("Selected index:", optimized_response.index)
print(
    "Extracted output from selected index:", predicted_outputs[optimized_response.index]
)

print("=== Original Version ===")
original_client = completion
start_time_original = time.perf_counter()
original_response = original_client(
    model="mistral/ministral-3b-2410",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Select the best output from the given options and write the exact thing.",
        },
        {
            "role": "user",
            "content": f"Here are the options: {predicted_outputs}",
        },
        {
            "role": "user",
            "content": "Tell me about burj khalifa.",
        },
    ],
)
end_time_original = time.perf_counter()
print(f"Original time: {end_time_original - start_time_original}")
print(original_response.choices[0].message.content)

print("=== Results ===")
print("Optimized time:", end_time_optimized - start_time_optimized)
print("Original time:", end_time_original - start_time_original)
print(
    "We saved",
    (end_time_original - start_time_original)
    - (end_time_optimized - start_time_optimized),
    "seconds",
)
print(
    f"And the outputs are {'same' if predicted_outputs[optimized_response.index] == original_response.choices[0].message.content else 'different'}"
)
