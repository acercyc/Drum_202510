# The current approach

To measure sense of agency without oral report. 

1. Whether the action is controlled by external cues.
I assume that when the action is driven by cues, then it's passive and not controlled by internal processes. To measure this, 
we can compute transfer entropy from the cue signal to the EMG action signal. 
This lead to three possible outcomes:
- TE is high: the action is driven by the cue signal: passive, low sense of agency
- TE is low: the action is not driven by the cue signal: active, high sense of agency
- TE is low: Action is stochastic, and not driven by the cue signal: low sense of agency

Therefore, we need a second measure to distinguish between the last two cases.

2. Measure the onset time difference between the cue signal and the hit signal.
If the action is active, then the onset time of the action should be close to the onset time of the cue signal with regular inter-hit intervals.
If the action is stochastic, then the onset time difference of the action should be random with irregular inter-hit intervals.
