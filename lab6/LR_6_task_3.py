p_rain_yes = 3/9
p_high_yes = 3/9
p_strong_yes = 3/9
p_rain_no = 2/5
p_high_no = 4/5
p_strong_no = 3/5
p_yes = 9/14
p_no = 5/14
score_yes = p_rain_yes * p_high_yes * p_strong_yes * p_yes
score_no = p_rain_no * p_high_no * p_strong_no * p_no
print(f"Показник для 'Yes': {score_yes:.5f}")
print(f"Показник для 'No': {score_no:.5f}")
total_score = score_yes + score_no
final_p_yes = score_yes / total_score
final_p_no = score_no / total_score
print("\n--- Результати прогнозу ---")
print(f"Ймовірність, що матч відбудеться ('Yes'): {final_p_yes:.1%}")
print(f"Ймовірність, що матч не відбудеться ('No'): {final_p_no:.1%}")
if final_p_yes > final_p_no:
    print("\nВисновок: Модель прогнозує, що матч відбудеться.")
else:
    print("\nВисновок: Модель прогнозує, що матч НЕ відбудеться.")
