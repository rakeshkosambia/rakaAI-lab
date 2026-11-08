# Task planning

For a long job the agent writes a plan: a short list of steps. It marks one step in progress, finishes it, then moves on. The plan keeps the work on course, because the model can see what is done and what is left. When the conversation grows, context compaction may summarise away the early turns, so the plan is added back to the prompt each time instead of being trusted to survive. Only one step should be in progress at a time. A plan that is saved with the run state lets a restarted agent continue from the next step.
