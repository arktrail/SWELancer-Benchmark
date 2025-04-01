import asyncio
import functools
import os
import re
import shlex
import subprocess
import threading
import traceback

from contextlib import asynccontextmanager, AsyncExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from textwrap import dedent
from typing import (
    Any,
    AsyncGenerator,
    cast,
    ContextManager,
    Generator,
    Generic,
    TypeVar,
)

import chz
import tiktoken
from alcatraz.clusters.local import LocalConfig
from nanoeval.asyncio_utils import generator_with_cleanup
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerInterface,
    JupyterComputerInterface,
)
from nanoeval.solvers.computer_tasks.solver import (
    PythonCodingEval,
    PythonCodingSolver,
    strip_all_metadata,
)
from nanoeval.solvers.computer_tasks.steps import (
    FinalResult,
    FinalResultSuccessful,
    FinalResultWithException,
    Step,
)
from nanoeval.solvers.computer_tasks.task import ComputerTask, Grade
from nanoeval_alcatraz.alcatraz_computer_interface import AlcatrazComputerInterface
from nanoeval_alcatraz.task_to_alcatraz_config import task_to_alcatraz_config

from openai import AsyncOpenAI
from typing_extensions import override


client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
)


def count_tokens(messages: list[dict[str, Any]], model: str = "gpt-4") -> int:
    """Count the number of tokens in a list of messages."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0

    for message in messages:
        # Every message follows format: {"role": role, "content": content}
        num_tokens += 4  # Every message follows format: <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))

    return num_tokens


def trim_messages(
    messages: list[dict[str, Any]], max_tokens: int, model: str = "gpt-4"
) -> list[dict[str, Any]]:
    """Trim messages to fit within token limit by removing older messages."""
    while len(messages) > 1 and count_tokens(messages, model) > max_tokens:
        messages.pop(1)
    return messages


async def get_model_response(messages: list[dict[str, Any]]) -> str:
    messages = trim_messages(messages, 110000)

    chat_completion = await client.chat.completions.create(
        messages=messages, model="gpt-4o", max_tokens=16384  # type: ignore
    )
    return chat_completion.choices[0].message.content  # type: ignore


@chz.chz
class SimpleAgentSolver(PythonCodingSolver):
    name: str = "SimpleAgentSolver"
    model: str = "gpt-4o"

    def shortname(self) -> str:
        return "simple-solver"

    @asynccontextmanager
    async def _start_computer(
        self, task: ComputerTask
    ) -> AsyncGenerator[ComputerInterface, None]:
        # replace with LocalCluster semantics

        alcatraz_env = task_to_alcatraz_config(
            task, LocalConfig(pull_from_registry=False)
        )

        async with alcatraz_env.build() as cluster:
            yield AlcatrazComputerInterface(cluster_value=cluster)

    @override
    async def run(self, task: ComputerTask) -> AsyncGenerator[Step | FinalResult, None]:
        try:
            async with self._start_computer(task) as computer:
                print(computer)
                # 1. Run the task setup
                await task.setup(computer)

                # 2. Query the API / some agent
                messages = []
                for prompt_message in task.prompt:
                    messages.append(
                        {
                            "role": "user",
                            "content": str(prompt_message["content"]),  # type: ignore
                        }
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": """You are an expert software engineer maintaining the Expensify repository.
It has already been cloned, and is present in /app/expensify (your CWD).
You have been tasked with fixing the following issue:
{issue title}
{full HTML documentation of issue, including how to reproduce the bug, the actual vs. expected behavior, and which platforms are affected}

The reward for the correct fix is ${price}.

You do not need to push your changes to the repository, we will deal with this for you.
Do not attempt to run the app; simply make your changes and then end your turn.
Do not end your turn until you have submitted your code changes.
You absolutely must not leave action items / TODOs.
---
You will be provided with a user tool, which you can call by running "bash -i -c ’user-tool’" in the command line, which has a user open the app with your local changes and perform the actions specified in the bug report.
It could take up to a minute for the user to finish using the app.
When you call the user tool, when the user finishes /app/expensify/user_tool/ will be populated with folders called output_browser1 ... n for as many browsers as were used by the user during their test.

Inside each folder, you will find a file called trace.trace, which contains the trajectory the user took. Here are a few things to note:
- You will need to parse this file programmatically in order to open it.
- If a row has "type":"screencast-frame", then the "sha1" property will correspond to a .jpeg file in the resources folder, which is a screenshot of the browser at that time.
- If you want to view the page as HTML, rather than a JPEG, you can look for rows that are called "type":"frame-snapshot"; you can print them out to see the HTML at this point in time.
- You should review the rows with type "type" set to "log", "before", or "after" as they are important and show what actions are being taken.
- All of the rows are in order, so e.g. you can find an interesting log row and then look at the next row that is a frame-snapshot to see the HTML.

You should use this to help you iteratively debug and fix the issue. You can call it as often as you need to. When you call the user tool, you wait 90,000 milliseconds (e.g. by setting the timeout argument to 90000)!
""",
                    }
                )
                max_turns = 30
                print(messages)

                for remaining_turns in range(max_turns, 0, -1):
                    model_response = await get_model_response(messages)
                    print(model_response)

                    messages.append({"role": "assistant", "content": model_response})

                    execution_output = None

                    # Check for user-tool calls
                    if "<user-tool>" in model_response:
                        print("User tool called.")
                        await computer.check_shell_command(
                            "unzip -q -P 'secretpasswordhere' /app/tests.zip -d /app/tests"
                        )
                        execution_output = await computer.check_shell_command(
                            "bash -c -i 'user-tool'"
                        )
                        print("User tool ran, with output:")
                        print(execution_output)
                        execution_output = (
                            "User tool output is in /app/expensify/user_tool"
                        )
                        await computer.send_shell_command("rm -rf /app/tests")
                        await computer.send_shell_command("rm -rf /app/tests")
                    else:
                        # Extract Python code from the model response
                        python_blocks = re.findall(
                            r"```python\s*\n(.*?)\n```", model_response, re.DOTALL
                        )

                        if not python_blocks:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": "Warning: No Python code blocks were found in the response. Please include Python code in your replies.",
                                }
                            )
                            continue

                        code = dedent(python_blocks[0])

                        execution_output = await computer.send_shell_command(
                            f"python -c {shlex.quote(code)}"
                        )
                        execution_output = execution_output.output.decode(
                            "utf-8", errors="replace"
                        )

                    if model_response.lower() == "done":
                        print("Breaking because model is done!")
                        break

                    print(execution_output)

                    # Append the code and its output to the messages
                    messages.append(
                        {
                            "role": "user",
                            "content": f"{execution_output}\nTurns left: {remaining_turns - 1}",
                        }
                    )

                # 3. Grade and yield the final result
                grade = await task.grade(computer)
                yield FinalResultSuccessful(grade=grade)
        except Exception as e:
            print(f"Error: {e}")
            raise
            yield FinalResultSuccessful(
                grade=Grade(score=0, grader_log=f"Grading failed with error: {str(e)}")
            )
