from typing import Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field
import numpy as np
from typing import Any, Callable, List, Optional, cast
from langgraph.prebuilt import ToolNode


class LinearSolverInput(BaseModel):
    A: List[List[float]] = Field(description="Coefficient matrix used for solving the system of linear equations",
                          examples=[[8, -3, 2], [4, 11, -1], [6, 3, 12]])
    B: List[float] = Field(description="Right-hand side constant vector",
                          examples=[20, 33, 36])
    error_tolerance: Optional[float] = Field(description="The error tolerance for the Jacobi iterative method",
                                   examples=1e-3,
                                   default=1e-3)
    iterate_limit: Optional[int] = Field(description="The maximum number of iterations for the Jacobi iterative method",
                                      examples=100,
                                      default=100)
    class Config:
        arbitrary_types_allowed = True
        
def format_matrix(matrix, name, precision=4):
    formatted_str = f"{name} = [\n"
    for row in matrix:
        formatted_row = "  [" + "  ".join(f"{val:{precision+6}.{precision}f}" for val in row) + "]\n"
        formatted_str += formatted_row
    formatted_str += "]"
    return formatted_str
        
class Jacobi_iterater(BaseTool):
    name: str = "Jacobi_iterater"
    description: str  = "Solve a system of linear equations using the Jacobi iterative method"
    args_schema: Optional[ArgsSchema] = LinearSolverInput
    response_format: str = "content_and_artifact"
    
    def _run(self,
             A: List[List[float]], B: List[List[float]], error_tolerance: float=1e-3,iterate_limit: int=100,
             run_manager: Optional[CallbackManagerForToolRun] = None):
        
        A = np.array(A)
        B = np.array(B)
        
        message = ""
        message += """Step 1: Decompose the coefficient matrix A into a lower triangular matrix L, an upper triangular matrix U and a diagonal matrix D.\n"""
        L = np.tril(A, k=-1)
        U = np.triu(A, k=1)
        D = np.diag(np.diag(A))
        message += f""" 
        <step 1>
        Matrix L is the strictly lower triangular part of A.
        Matrix U is the strictly upper triangular matrix.
        Matrix D is the diagonal matrix of A.
        Thus we have:
            {format_matrix(L, 'L')}, 
            {format_matrix(U, 'U')},
            {format_matrix(D, 'D')}
        </step 1>
        """
        
        message += """Step 2: Compute the iteration matrix B_jacobi and the constant vector f_jacobi.\n"""
        inv_D = np.linalg.inv(D)
        B_jacobi = -inv_D @ (L + U)
        f_jacobi = inv_D @ B
        message += f""" 
        <step 2>
        The Jacobi iteration formula is given by: x_new = B_jacobi @ x + f_jacobi
        where B_jacobi = D^(-1) @ (L + U) and f_jacobi = D^(-1) @ B
        Calculate explicitly:
        {format_matrix(inv_D, 'D^(-1)')}
        Then we have:
        {format_matrix(B_jacobi, 'B_jacobi')}
        f_jacobi = {f_jacobi}
        </step 2>
        """
        
        message += """Step 3: Initialize the solution vector x and the iteration counter k.\n"""
        x = np.zeros_like(B)
        k = 0
        error = float('inf')
        message += f""" 
        <step 3>
        Initialize the solution vector x = {x} and the iteration counter k = {k}
        </step 3>
        """
        
        message += """Step 4: Iterate the Jacobi formula until the error is less than the tolerance or over iterate limitation.\n
        <step 4>"""
        
        while k < iterate_limit and error > error_tolerance:
            x_new = B_jacobi @ x + f_jacobi
            error = np.linalg.norm(x_new - x)
            k += 1
            x = x_new
            message += f""" 
            <iteration {k}>
            x = {x}
            error = {error}
            </iteration {k}>
            """
            
        message += f""" 
        The solution vector x is {x}
        The error is {error}
        The iteration counter k is {k}
        </step 4>
        """
        
        return message, x
    
class Gauss_Seidel_Iterater(BaseTool):
    name: str = "Gauss_Seidel_Iterater"
    description: str  = "Solve a system of linear equations using the Gauss-Seidel iterative method"
    args_schema: Optional[ArgsSchema] = LinearSolverInput
    response_format: str = "content_and_artifact"
    
    def _run(self,
             A: List[List[float]], B: List[List[float]], error_tolerance: float=1e-3,iterate_limit: int=100,
             run_manager: Optional[CallbackManagerForToolRun] = None):
        
        A = np.array(A)
        B = np.array(B)
        
        message = ""
        message += """Step 1: Decompose the coefficient matrix A into a lower triangular matrix L, an upper triangular matrix U and a diagonal matrix D.\n"""
        L = np.tril(A, k=-1)
        U = np.triu(A, k=1)
        D = np.diag(np.diag(A))
        message += f""" 
        <step 1>
        Matrix L is the strictly lower triangular part of A.
        Matrix U is the strictly upper triangular matrix.
        Matrix D is the diagonal matrix of A.
        Thus we have:
            {format_matrix(L, 'L')}, 
            {format_matrix(U, 'U')},
            {format_matrix(D, 'D')}
        </step 1>
        """
        
        message += """Step 2: Compute the iteration matrix B_gauss_seidel and the constant vector f_gauss_seidel.\n"""
        inv_D_plus_L = np.linalg.inv(D + L)
        B_gauss_seidel = -inv_D_plus_L @ U
        f_gauss_seidel = inv_D_plus_L @ B
        message += f""" 
        <step 2>
        The Gauss-Seidel iteration formula is given by: x_new = B_gauss_seidel @ x + f_gauss_seidel
        where B_gauss_seidel = -(D + L)^(-1) @ U and f_gauss_seidel = (D + L)^(-1) @ B
        Calculate explicitly:
        {format_matrix(inv_D_plus_L, '(D + L)^(-1)')}
        Then we have:
        {format_matrix(B_gauss_seidel, 'B_gauss_seidel')}
        f_gauss_seidel = {f_gauss_seidel}
        </step 2>
        """
        
        message += """Step 3: Initialize the solution vector x and the iteration counter k.\n"""
        x = np.zeros_like(B)
        k = 0
        error = float('inf')
        message += f"""
        <step 3>
        Initialize the solution vector x = {x} and the iteration counter k = {k}
        </step 3>
        """
        
        message += """Step 4: Iterate the Gauss-Seidel formula until the error is less than the tolerance or over iterate limitation.\n
        <step 4>"""
        
        while k < iterate_limit and error > error_tolerance:
            x_new = B_gauss_seidel @ x + f_gauss_seidel
            error = np.linalg.norm(x_new - x)
            k += 1
            x = x_new
            message += f""" 
            <iteration {k}>
            x = {x}
            error = {error}
            </iteration {k}>
            """
            
        message += f"""
        The solution vector x is {x}
        The error is {error}
        The iteration counter k is {k}
        </step 4>
        """
        
        return message, x
    
TOOLS: List[Callable[..., Any]] = [Jacobi_iterater(), Gauss_Seidel_Iterater()]