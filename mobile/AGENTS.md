Strongly discourage the use of any. If any seems unavoidable, ask the user first and explain why it is needed before introducing it.

Default to required properties in TypeScript types/interfaces. Only mark a property as optional when the user explicitly says it is optional. If there is a strong reason to make it optional (e.g., staged hydration or partial API responses), ask the user first and explain why you think optionality is justified. We prioritize certainty and consistency from the type system to reduce optional chaining and runtime uncertainty.

Always run npm run typecheck after making code changes to ensure type checks pass.
