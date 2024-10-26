import time, datetime
import numpy as np

from commons.algorithms.algorithms import Algorithms
from tensorboardX import SummaryWriter
from pathlib import Path

class Logger:
    def __init__(self, directory):
        save_dir = Path(directory.format("boards"))
        save_dir.mkdir(parents=True, exist_ok=True)

        self.writter = SummaryWriter(save_dir)
        self.record_time = time.time()
        
        self.window_size = 100

        self.rewards = []
        self.rewards_moving = []
        self.rewards_accumulated = []

        self.last_q_values = np.array([]);

        self.last_step = 0

    def report_evaluation(self, episode:int, avg_reward: int, best_score: int):
        print(
            f"Evaluate/Reward avg {avg_reward} - "
            f"Evaluate/Best score {best_score}"
        )

        self.writter.add_scalar("4.Evaluate/Reward avg", avg_reward, episode)
        self.writter.add_scalar("4.Evaluate/Best score", best_score, episode)

    def report_train(self, alg: Algorithms, episode:int, episody_reward:float):
        self.rewards.append(episody_reward)
        if len(self.rewards) >= self.window_size:
            self.rewards_moving.append(sum(self.rewards[-self.window_size:]) / self.window_size)
        else:
            self.rewards_moving.append(sum(self.rewards) / len(self.rewards))

        last_reward = self.rewards_accumulated[-1] if self.rewards_accumulated else 0
        self.rewards_accumulated.append(last_reward + episody_reward)

        reward_rate_of_change = 0
        if episode >= self.window_size : 
            reward_rate_of_change =  (self.rewards[episode] - self.rewards[episode - self.window_size]) / self.window_size

        standar_desviation = np.std(self.rewards_moving)
        interquartile_range = np.percentile(self.rewards_moving, 75) - np.percentile(self.rewards_moving, 25)

        if(alg.q_values.nbytes != 0):
            if(self.last_q_values.nbytes == 0):
                self.last_q_values = np.zeros_like(alg.q_values)    

            variation = np.mean(np.abs(alg.q_values - self.last_q_values))
            self.last_q_values = alg.q_values
        else:
            variation = 0

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        length = alg.steps - self.last_step
        self.last_step = alg.steps

        print(
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')} - "
            f"Delta {time_since_last_record} - "
            f"Episode {episode} - "
            f"Length {length} - "
            f"Reward {episody_reward} - "
            f"Accumulated {self.rewards_accumulated[episode]} - "
            f"Epsilon {alg.epsilon}"
        )

        if(hasattr(alg, "q_max") and hasattr(alg, "td_error") and hasattr(alg, "v_avg")):
            print(
                f"General/Q Max {alg.q_max} - "
                f"General/TD Error {alg.td_error} - "
                f"General/V Avg {alg.v_avg}"
            )

        print(
            f"Convergence/Moving reward {self.rewards_accumulated[episode]} - "
            f"Convergence/Rate change {reward_rate_of_change} "
        )

        print(
            f"Stability/Standar desviation {standar_desviation} - "
            f"Stability/Variation {variation} - "
            f"Stability/Interquartile range {interquartile_range}"
        )

        self.writter.add_scalar("0.General/Epsilon", alg.epsilon, episode)
        if(hasattr(alg, "q_max") and hasattr(alg, "td_error") and hasattr(alg, "v_avg")):
            self.writter.add_scalar("0.General/Q Max", alg.q_max, episode)
            self.writter.add_scalar("0.General/TD Error", alg.td_error, episode)
            self.writter.add_scalar("0.General/V Avg", alg.v_avg, episode)
        
        self.writter.add_scalar("1.Episody/Length", length, episode)
        self.writter.add_scalar("1.Episody/Reward", episody_reward, episode)
        self.writter.add_scalar("1.Episody/Total", self.rewards_accumulated[episode], episode)

        self.writter.add_scalar("2.Convergence/Moving reward", self.rewards_moving[episode], episode)
        self.writter.add_scalar("2.Convergence/Rate change", reward_rate_of_change, episode)

        self.writter.add_scalar("3.Stability/Standar desviation", standar_desviation, episode)
        self.writter.add_scalar("3.Stability/Variation", variation, episode)
        self.writter.add_scalar("3.Stability/Interquartile range", interquartile_range, episode)

    def closeWritter(self):
        self.writter.close()



'''
def report_train(self, alg: Algorithms, episode: int, episody_reward: float):
    # Almacena las recompensas por episodio
    self.rewards.append(episody_reward)
    
    # Calcula las recompensas móviles con una ventana de tamaño window_size
    if len(self.rewards) >= self.window_size:
        moving_avg = sum(self.rewards[-self.window_size:]) / self.window_size
    else:
        moving_avg = sum(self.rewards) / len(self.rewards)  # Maneja el caso donde no hay suficientes episodios aún
    self.rewards_moving.append(moving_avg)

    # Calcula la recompensa acumulada
    last_reward = self.rewards_accumulated[-1] if self.rewards_accumulated else 0
    self.rewards_accumulated.append(last_reward + episody_reward)

    # Cálculo de tasa de cambio en las recompensas (Rate of change)
    reward_rate_of_change = 0
    if episode >= self.window_size: 
        reward_rate_of_change = (self.rewards[episode] - self.rewards[episode - self.window_size]) / self.window_size

    # Cálculo de métricas estadísticas: desviación estándar, media, etc.
    standar_desviation = np.std(self.rewards_moving)
    mean = np.mean(self.rewards_moving)
    interquartile_range = np.percentile(self.rewards_moving, 75) - np.percentile(self.rewards_moving, 25)

    # Actualización de la métrica "Variation" basada en el cambio absoluto de los valores Q
    if hasattr(alg, "q_values"):  # Asegurarse de que el algoritmo tiene acceso a los valores Q
        # Guardar los valores Q actuales
        current_q_values = np.array(alg.q_values)  # Suponiendo que 'alg.q_values' es una lista o array de los valores Q actuales
        
        # Si es el primer episodio, inicializamos el último valor de Q como 0s
        if not hasattr(self, 'last_q_values'):
            self.last_q_values = np.zeros_like(current_q_values)

        # Cálculo del cambio absoluto de los valores Q entre episodios
        q_variation = np.mean(np.abs(current_q_values - self.last_q_values))

        # Actualizamos el último valor de Q para la próxima iteración
        self.last_q_values = current_q_values
    else:
        q_variation = 0  # Si no se tienen los valores Q, la variación es 0

    # Tiempo transcurrido desde el último reporte
    last_record_time = self.record_time
    self.record_time = time.time()
    time_since_last_record = np.round(self.record_time - last_record_time, 3)

    # Longitud del episodio
    length = alg.steps - self.last_step
    self.last_step = alg.steps

    # Impresión de métricas de entrenamiento
    print(
        f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')} - "
        f"Delta {time_since_last_record} - "
        f"Episode {episode} - "
        f"Length {length} - "
        f"Reward {episody_reward} - "
        f"Accumulated {self.rewards_accumulated[episode]} - "
        f"Epsilon {alg.epsilon}"
    )

    # Impresión de métricas de Q-Learning si están disponibles
    if hasattr(alg, "q_max") and hasattr(alg, "td_error") and hasattr(alg, "v_avg"):
        print(
            f"General/Q Max {alg.q_max} - "
            f"General/TD Error {alg.td_error} - "
            f"General/V Avg {alg.v_avg}"
        )

    # Métricas de convergencia
    print(
        f"Convergence/Moving reward {self.rewards_accumulated[episode]} - "
        f"Convergence/Rate change {reward_rate_of_change} "
    )

    # Métricas de estabilidad
    print(
        f"Stability/Standar desviation {standar_desviation} - "
        f"Stability/Q Value Variation {q_variation} - "  # Mostramos la variación de los valores Q
        f"Stability/Interquartile range {interquartile_range}"
    )

    # Almacenamiento de métricas en TensorBoard
    self.writter.add_scalar("0.General/Epsilon", alg.epsilon, episode)
    if hasattr(alg, "q_max") and hasattr(alg, "td_error") and hasattr(alg, "v_avg"):
        self.writter.add_scalar("0.General/Q Max", alg.q_max, episode)
        self.writter.add_scalar("0.General/TD Error", alg.td_error, episode)
        self.writter.add_scalar("0.General/V Avg", alg.v_avg, episode)

    # Métricas del episodio
    self.writter.add_scalar("1.Episody/Length", length, episode)
    self.writter.add_scalar("1.Episody/Reward", episody_reward, episode)
    self.writter.add_scalar("1.Episody/Total", self.rewards_accumulated[episode], episode)

    # Métricas de convergencia
    if len(self.rewards_moving) > episode:  # Asegúrate de no pasar un índice inválido
        self.writter.add_scalar("2.Convergence/Moving reward", self.rewards_moving[episode], episode)
    self.writter.add_scalar("2.Convergence/Rate change", reward_rate_of_change, episode)

    # Métricas de estabilidad
    self.writter.add_scalar("3.Stability/Standar desviation", standar_desviation, episode)
    self.writter.add_scalar("3.Stability/Q Value Variation", q_variation, episode)  # Almacenamos la variación de Q
    self.writter.add_scalar("3.Stability/Interquartile range", interquartile_range, episode)
'''        