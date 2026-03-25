Decision: Create a decisions directory for logging significant design decisions

Options considered:
- No dedicated directory: Log decisions in general documentation or commit messages
- Use a different format: Plain text files or integrate into existing READMEs
- Database or tool-based: Use a specialized tool for decision tracking

Chosen: Create a dedicated decisions directory with markdown files

Rationale: A dedicated directory ensures decisions are centralized and easily accessible for the project report. Markdown files provide readable formatting and can be version controlled alongside the code. This approach is simple, requires no additional tools, and follows common practices in software projects.

Trade-offs: Requires discipline to maintain and update the files; potential for files to become outdated if not kept current. However, the benefits of having documented decisions outweigh the minimal overhead, especially if utilising AI to help automate documentation.