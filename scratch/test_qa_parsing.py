import sys
import os

# Add parent directory to sys.path so we can import core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.qa_generator import QAGenerator, parse_timestamp_to_seconds

def test_timestamp_parsing():
    assert parse_timestamp_to_seconds("00:00:05") == 5.0
    assert parse_timestamp_to_seconds("00:02") == 2.0
    assert parse_timestamp_to_seconds("01:05:30.5") == 3930.5
    print("[+] Timestamp parsing test passed!")

def test_custom_qa_generation():
    # Setup mock processed frames
    mock_frames = [
        {
            "frame_idx": 0,
            "timestamp": 0.0,
            "detections": [{"label": "person", "box": [10, 10, 50, 50], "score": 0.9}],
            "blur_var": 120.0,
            "brightness": 100.0,
        },
        {
            "frame_idx": 5,
            "timestamp": 5.0,
            "detections": [
                {"label": "person", "box": [12, 10, 52, 50], "score": 0.9},
                {"label": "car", "box": [100, 100, 200, 200], "score": 0.95}
            ],
            "blur_var": 110.0,
            "brightness": 95.0,
        },
        {
            "frame_idx": 10,
            "timestamp": 10.0,
            "detections": [{"label": "car", "box": [105, 100, 205, 200], "score": 0.95}],
            "blur_var": 130.0,
            "brightness": 102.0,
        }
    ]

    captions = "A mock video showing a pedestrian walking and a car driving by."
    predefined_questions = (
        "How many cars are visible between 00:00:02 and 00:00:10?\n"
        "Is there a person present in the video?\n"
        "Is there a dog present after 00:00:02?"
    )

    generator = QAGenerator(
        filename="mock_run.mp4",
        processed_frames=mock_frames,
        duration=10.0,
        qa_categories=["user_queries"],
        captions=captions,
        example_questions=predefined_questions
    )

    import pprint
    print("TRACKS:")
    pprint.pprint(generator.track_objects())
    
    qa_by_category = generator.generate_qa_pairs()
    assert "user_queries" in qa_by_category
    user_qa = qa_by_category["user_queries"]

    pprint.pprint(user_qa)

    # Expected: 4 items (1 caption description + 3 predefined questions)
    assert len(user_qa) == 4

    # Check Caption QA
    assert user_qa[0]["Question"] == "What is the context of this video?"
    assert user_qa[0]["Answer"] == captions
    assert user_qa[0]["Answer format"] == "open-ended"

    # Check question 1: How many cars are visible between 00:00:02 and 00:00:10?
    assert user_qa[1]["Question"] == "How many cars are visible between 00:00:02 and 00:00:10?"
    assert user_qa[1]["Answer"] == "2" # 2 car tracks (due to large sampling interval)
    assert user_qa[1]["Answer format"] == "open-ended"

    # Check question 2: Is there a person present in the video?
    assert user_qa[2]["Question"] == "Is there a person present in the video?"
    assert user_qa[2]["Answer"] == "yes"
    assert user_qa[2]["Answer format"] == "yes-no"

    # Check question 3: Is there a dog present after 00:00:02?
    assert user_qa[3]["Question"] == "Is there a dog present after 00:00:02?"
    assert user_qa[3]["Answer"] == "no"
    assert user_qa[3]["Answer format"] == "yes-no"

    print("[+] Custom QA generation and evaluation tests passed!")

if __name__ == "__main__":
    test_timestamp_parsing()
    test_custom_qa_generation()
    print("[+] All automated tests run and passed successfully!")
