

struct TransportationMDP
    N::Int32
    discount::Float64
end

function isEnd(mdp::TransportationMDP, state)
    return state == mdp.N
end

function actions(mdp::TransportationMDP, state)
    result = String[]
    if state +1 <= mdp.N
        push!(result, "walk")
    end
    if 2*state <= mdp.N
        push!(result, "tram")
    end
    return result
end
    
function succProbReward(state, action)
    result = []
    if action == "walk"
        push!(result, [state + 1, 1, -1])
    elseif action == "tram"
        push!(result, [2 * state, .5, -2])
        push!(result, [state, .5, -2])
    end
    return result
end

function states(mdp::TransportationMDP)
    return range(1, stop=mdp.N)
end


function valueIteration(mdp::TransportationMDP)
    # Initialize value and policy arrays
    value =  Array{Float64}(undef, length(states(mdp)))
    p = Array{String}(undef, length(states(mdp)))
    for (i, state) in enumerate(states(mdp))
        value[i] = 0
    end
    # Q Function
    function Q(mdp::TransportationMDP, state, action)
        q = sum(prob * (reward + mdp.discount * value[Int(newState)]) for (newState, prob, reward) in succProbReward(state, action))
        return q
    end
    # This should be until convergence
    for j in 1:10
        vNew =  Array{Float64}(undef, length(states(mdp)))
        for (i, state) in enumerate(states(mdp))
            if isEnd(mdp, state)
                vNew[i] = 0
            else 
                vNew[i] = maximum([Q(mdp, state, action) for action in actions(mdp, state)])
            end
        end
        value = vNew

        # Get policy
        for (i, state) in enumerate(states(mdp))
            if isEnd(mdp, state)
                p[i] = "None"
            else
                p[i] = maximum((Q(mdp, state, action), action) for action in actions(mdp, state))[2]
            end
        end
    end
    println("S", "\t", "V(s)", "\t", "Pi")
    for (i, state) in enumerate(states(mdp))
        println(state, "\t", value[i], "\t", p[i])
    end
end

test = TransportationMDP(10, 1.0)

# Runs MDP and outputs Optimal Policy with V(S)
valueIteration(test)