from typing import Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field
import numpy as np
from typing import Any, Callable, List, Optional, cast, Union
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
        
class SORSolverInput(BaseModel):
    A: List[List[float]] = Field(description="Coefficient matrix used for solving the system of linear equations",
                          examples=[[8, -3, 2], [4, 11, -1], [6, 3, 12]])
    B: List[float] = Field(description="Right-hand side constant vector",
                          examples=[20, 33, 36])
    omega: Optional[float] = Field(description="The relaxation factor for the SOR iterative method",
                                   examples=1.5,
                                   default=1.5)
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

# 判断是否对角占优
def is_diagonally_dominant(A:np.ndarray) -> bool:
    """
    Judge whether the matrix A is diagonally dominant.
    """
    D = np.diag(np.abs(A))
    S = np.sum(np.abs(A), axis=1) - D
    
    if (D > S).all():
        return True
    else:
        return False
    
# 是否对称正定
def is_symmetric_positive_definite(A:np.ndarray) -> bool:
    """
    Judge whether the matrix A is symmetric positive definite.
    """
    if (A == A.T).all() and np.all(np.linalg.eigvals(A) > 0):
        return True
    else:
        return False
    
# 谱半径
def spectral_radius(A:np.ndarray) -> float:
    """
    Calculate the spectral radius of the matrix A.
    """
    return np.max(np.abs(np.linalg.eigvals(A)))

# 判断是否收敛
def is_converged(name:str, A:np.ndarray, B:np.ndarray) -> Union[bool, str]:
    """
    Judge whether the iterative method converges.
    """
    
    # 1. is diagonally dominant
    if is_diagonally_dominant(A):
        return True, f"The matrix is diagonally dominant. So the {name} iterative method converge."
    
    # 2. is symmetric positive definite
    if is_symmetric_positive_definite(A):
        if name == "Gauss_Seidel_Iterater" or name == "SOR_Iterater":
            return True, f"The matrix is symmetric positive definite. So the {name} iterative method converge."
        
    # 3. spectral radius
    r = spectral_radius(B)
    if r < 1:
        return True, f"The spectral radius of the iteration matrix is less than 1. Which equal to {r}, So the {name} iterative method converge."
    
    return False, f"After checking the diagonally dominant, symmetric positive definite and spectral radius({r}, where r > 1), the {name} iterative method does not converge."
    
    
        
class Jacobi_iterater(BaseTool):
    name: str = "Jacobi_iterater"
    description: str  = "Solve a system of linear equations using the Jacobi iterative method, it can also be used to check the convergence of the Jacobi iterative method."
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
        
        message += """Step 3 Check whether the Jacobi iterative method converges.\n"""
        is_converged_jacobi, reason = is_converged("Jacobi", A, B_jacobi)
        message += f"""
        <step 3>
        {reason}
        </step 3>
        """
        
        if is_converged_jacobi:
            message += """Step 4: Initialize the solution vector x and the iteration counter k.\n"""
            x = np.zeros_like(B)
            k = 0
            error = float('inf')
            message += f""" 
            <step 4>
            Initialize the solution vector x = {x} and the iteration counter k = {k}
            </step 4>
            """
            
            message += """Step 5: Iterate the Jacobi formula until the error is less than the tolerance or over iterate limitation.\n
            <step 5>"""
            
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
            </step 5>
            """
            
            return message, x
        else:
            return message, None
    
class Gauss_Seidel_Iterater(BaseTool):
    name: str = "Gauss_Seidel_Iterater"
    description: str  = "Solve a system of linear equations using the Gauss-Seidel iterative method, it can also be used to check the convergence of the Gauss-Seidel iterative method."
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
        
        message += """Step 3 Check whether the Gauss-Seidel iterative method converges.\n"""
        is_converged_gauss_seidel, reason = is_converged("Gauss-Seidel", A, B_gauss_seidel)
        message += f"""
        <step 3>
        {reason}
        </step 3>
        """
        
        if is_converged_gauss_seidel:
            message += """Step 4: Initialize the solution vector x and the iteration counter k.\n"""
            x = np.zeros_like(B)
            k = 0
            error = float('inf')
            message += f"""
            <step 4>
            Initialize the solution vector x = {x} and the iteration counter k = {k}
            </step 4>
            """
            
            message += """Step 5: Iterate the Gauss-Seidel formula until the error is less than the tolerance or over iterate limitation.\n
            <step 5>"""
            
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
            </step 5>
            """
            
            return message, x
        else:
            return message, None
        
class SOR_Iterater(BaseTool):
    name: str = "SOR_Iterater"
    description: str  = "Solve a system of linear equations using the SOR iterative method, it can also be used to check the convergence of the SOR iterative method."
    args_schema: Optional[ArgsSchema] = SORSolverInput
    response_format: str = "content_and_artifact"
    
    def _run(self,
             A: List[List[float]], B: List[List[float]], omega: float=1.5, error_tolerance: float=1e-3,iterate_limit: int=100,
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
        
        message += """Step 2: Compute the iteration matrix B_sor and the constant vector f_sor.\n"""
        inv_D_plus_L = np.linalg.inv(D + omega * L)
        B_sor = inv_D_plus_L @ ((1 - omega) * D - omega * U)
        f_sor = omega * inv_D_plus_L @ B
        
        message += f"""
        <step 2>
        The SOR iteration formula is given by: x_new = B_sor @ x + f_sor
        where B_sor = (D + omega * L)^(-1) @ ((1 - omega) * D - omega * U) and f_sor = omega * (D + omega * L)^(-1) @ B
        Calculate explicitly:
        {format_matrix(inv_D_plus_L, '(D + omega * L)^(-1)')}
        Then we have:
        {format_matrix(B_sor, 'B_sor')}
        f_sor = {f_sor}
        </step 2>
        """
        
        message += """Step 3 Check whether the SOR iterative method converges.\n"""
        is_converged_sor, reason = is_converged("SOR", A, B_sor)
        message += f"""
        <step 3>
        {reason}
        </step 3>
        """
        
        if is_converged_sor:
            message += """Step 4: Initialize the solution vector x and the iteration counter k.\n"""
            x = np.zeros_like(B)
            k = 0
            error = float('inf')
            message += f"""
            <step 4>
            Initialize the solution vector x = {x} and the iteration counter k = {k}
            </step 4>
            """
            
            message += """Step 5: Iterate the SOR formula until the error is less than the tolerance or over iterate limitation.\n
            <step 5>"""
            
            while k < iterate_limit and error > error_tolerance:
                x_new = B_sor @ x + f_sor
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
            </step 5>
            """
            
            return message, x
        else:
            return message, None
        
        
        
                
    
TOOLS: List[Callable[..., Any]] = [Jacobi_iterater(), Gauss_Seidel_Iterater(), SOR_Iterater()]