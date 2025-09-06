
import sys 
sys.path.append("../")
from utils import get_center_of_bbox, measure_distance



class PlayerBallAssigner :
    def __init__(self):
        self.max_player_ball_distance = 70 # in pixels, max distance between player and ball to consider the player as in possession of the ball    
    
    def assign_ball_to_players(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = float('inf')
        player_in_possession = None

        for player_id, player in players.items():
            player_bbox = player['bbox']

            # left foot of player and ball
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            # right foot of player and ball 
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position) 
            distance = min(distance_left, distance_right)
            
            if distance < self.max_player_ball_distance and distance < minimum_distance:
                minimum_distance = distance
                player_in_possession = player_id
                
        return player_in_possession
            
            
