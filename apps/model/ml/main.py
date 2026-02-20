from ml_model import CropRecommender

if __name__ == "__main__":
    print("\nCrop Recommendation System")
    print("-" * 40)
    recommender = CropRecommender()
    try:
        N = float(input("Nitrogen (N): "))
        P = float(input("Phosphorus (P): "))
        K = float(input("Potassium (K): "))
        temperature = float(input("Temperature (°C): "))
        humidity = float(input("Humidity (%): "))
        ph = float(input("pH value: "))
        rainfall = float(input("Rainfall (mm): "))
        user_input = [N, P, K, temperature, humidity, ph, rainfall]
        crop, confidence, alternatives = recommender.recommend_crop_with_alternatives(user_input)
        print("\nRecommended Crop:")
        print(f"- {crop}")
        print(f"Confidence: {confidence * 100:.2f}%")
        print("\nOther Suitable Options:")
        if alternatives:
            for i, (alt_crop, alt_conf) in enumerate(alternatives, start=1):
                print(f"{i}. {alt_crop} - {alt_conf * 100:.2f}%")
        else:
            print("No strong alternative crops found")
    except Exception:
        print("Invalid input! Please enter numeric values only")