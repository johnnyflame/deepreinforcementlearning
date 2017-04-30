import gym
import universe
import random

#reinforcement learning step
def determine_turn(turn,observation_n,j,total_sum,prev_total_sum,reward_n):
    #for every 15 iterations, sum the total observation, and take the average value.
    # if the value is less than 0, change the direction of turn.

    if(j >= 15):
        if(total_sum / j) == 0:
            turn = True
        else:
            turn = False

        total_sum = 0
        j = 0
        prev_total_sum = total_sum
        total_sum = 0
    else:
        turn = False

    if(observation_n != None):

        j+=1
        total_sum += reward_n

    return(turn,j,total_sum,prev_total_sum)



def main():

    #init environment
    env = gym.make("flashgames.CoasterRacer-v0")

    #Important if this is the first time the env is run.
    env.configure(remotes=1)

    #client is the agent
    #remote is the (local) docker container with the environment

    observation_n = env.reset()

    #init variables

    #num of game iterations
    n = 0
    j = 0

    #sum of observations
    total_sum = 0
    prev_total_sum = 0
    turn = False

    #define our turns or keyboard actions
    left = [('KeyEvent','ArrowUp',True),('KeyEvent','ArrowLeft',True),('KeyEvent','ArrowRight',False)]
    right = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True)]
    forward = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)]
    noop = [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)]

    #main logic

    while True:
        #increment counter per iteration
        n+=1

        if(n > 1):
            #Check if turn is required after at least one iteration
            if(observation_n[0] != None):
                #store the reward in the previous score
                prev_score = reward_n[0]

                #Should we turn?
                if(turn):
                    #pick a random event
                    event = random.choice([left,right])

                    #perform an action
                    action_n = [event for ob in observation_n]
                    #set turn to False
                    turn = False

        elif(~turn):
            action_n = [forward for ob in observation_n]


        if(observation_n[0] != None):
            turn,j,total_sum,prev_total_sum = determine_turn(turn,observation_n[0],j,total_sum,prev_total_sum,reward_n[0])


        observation_n,reward_n,done_n,info = env.step(action_n)
        env.render()

if __name__ == '__main__':
    main()

