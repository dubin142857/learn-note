
static char help[] = "Solves a linear system in parallel with KSP and DM.\n\
Compare this to ex2 which solves the same problem without a DM.\n\n";

/*T
   Concepts: KSP^basic parallel example;
   Concepts: KSP^Laplacian, 2d
   Concepts: DM^using distributed arrays;
   Concepts: Laplacian, 2d
   Processors: n
T*/

/*
  Include "petscdmda.h" so that we can use distributed arrays (DMDAs).
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include <petscdmda.h>
#include <petscksp.h>

PetscErrorCode SolutionSaveASCII(Vec x, const char filename[]);
PetscErrorCode SolutionSaveBinary(Vec x, const char filename[]);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DM             da;            /* distributed array */
  Vec            x,b,u;         /* approx solution, RHS, exact solution */
  Mat            A;             /* linear system matrix */
  KSP            ksp;           /* linear solver context */
  PetscRandom    rctx;          /* random number generator context */
  PetscReal      norm;          /* norm of solution error */
  PetscInt       i,j,its;
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;
  PetscLogStage  stage;
  DMDALocalInfo  info;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);

  /*
     Create distributed array to handle parallel distribution.
     The problem size will default to 8 by 7, but this can be
     changed using -da_grid_x M -da_grid_y N
  */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,-8,-7,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create parallel matrix preallocated according to the DMDA, format AIJ by default.
     To use symmetric storage, run with -dm_mat_type sbaij -mat_ignore_lower_triangular
  */
  ierr = DMCreateMatrix(da,MATAIJ,&A);CHKERRQ(ierr);

  /*
     Set matrix elements for the 2-D, five-point stencil in parallel.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly).
      - Rows and columns are specified by the stencil
      - Entries are normalized for a domain [0,1]x[0,1]
   */
  ierr = PetscLogStageRegister("Assembly", &stage);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  PetscReal   hx  = 1./(info.mx + 1),hy = 1./(info.my + 1);
  MatStencil  row = {0},col[5] = {{0}};
  PetscScalar v[5];
  PetscInt    ncols = 0;
  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) {
      ncols = 0;
      row.j        = j; row.i = i;
      col[ncols].j = j; col[ncols].i = i; v[ncols++] = 2*(hx/hy + hy/hx);
      /* boundaries */
      if (i>0)         {col[ncols].j = j;   col[ncols].i = i-1; v[ncols++] = -hy/hx;}
      if (i<info.mx-1) {col[ncols].j = j;   col[ncols].i = i+1; v[ncols++] = -hy/hx;}
      if (j>0)         {col[ncols].j = j-1; col[ncols].i = i;   v[ncols++] = -hx/hy;}
      if (j<info.my-1) {col[ncols].j = j+1; col[ncols].i = i;   v[ncols++] = -hx/hy;}
      ierr = MatSetValuesStencil(A,1,&row,ncols,col,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  /*
     Create parallel vectors compatible with the DMDA.
  */
  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&x);CHKERRQ(ierr);

  /*
     Set exact solution; then compute right-hand-side vector.
     By default we use an exact solution of a vector with all
     elements of 1.0;  Alternatively, using the runtime option
     -random_sol forms a solution vector with random components.
  */
  /*
  ierr = PetscOptionsGetBool(NULL,"-random_exact_sol",&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
    ierr = VecSetRandom(u,rctx);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  } else {
    ierr = VecSet(u,1.);CHKERRQ(ierr);
  }
  ierr = MatMult(A,u,b);CHKERRQ(ierr);
  */
  PetscScalar **uu, **bb, xi, yj;
  ierr = DMDAVecGetArray(da, u, &uu); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, b, &bb); CHKERRQ(ierr);
  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) {
      xi = (PetscReal)(i+1) * hx;
      yj = (PetscReal)(j+1) * hy;
      uu[j][i] = sin(2.*PETSC_PI*xi) * sin(2.*PETSC_PI*yj);
      bb[j][i] = 8. * PETSC_PI * PETSC_PI * sin(2.*PETSC_PI*xi) * sin(2.*PETSC_PI*yj) * hx * hy;
    }
  }
  ierr = DMDAVecRestoreArray(da, u, &uu); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, b, &bb); CHKERRQ(ierr);

  /*
     View the exact solution vector if desired
  */
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,"-view_exact_sol",&flg,NULL);CHKERRQ(ierr);
  if (flg) {ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create linear solver context
  */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  ierr = KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
  */
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscPrintf(PETSC_COMM_WORLD, "\n========== my Poisson problem ==========\n"); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "---------- KSP iterations begin ----------\n"); CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "---------- KSP iterations end ----------\n");

  PetscInt save_type = 0;  // data save format: 0 -- don't save,  1 -- save in ASCII,  2 -- save in binary
  ierr = PetscOptionsGetInt(NULL, "-save_type", &save_type, NULL); CHKERRQ(ierr);
  if (save_type != 0)
  {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Save solution to a file.\n");
    if (save_type == 1) { ierr = SolutionSaveASCII(x, "data/solution.txt"); CHKERRQ(ierr);}
    if (save_type == 2) { ierr = SolutionSaveBinary(x, "data/solution.bin"); CHKERRQ(ierr);}
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Check the error
  */
  ierr = VecAXPY(x,-1.,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);

  /*
     Print convergence information.  PetscPrintf() produces a single
     print statement from all processes that share a communicator.
     An alternative is PetscFPrintf(), which prints to a file.
  */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %G iterations %D\n",norm/sqrt((PetscReal)(info.mx*info.my)),its);CHKERRQ(ierr);

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary).
  */
  ierr = PetscFinalize();
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "SolutionSaveASCII"
PetscErrorCode SolutionSaveASCII(Vec x, const char filename[])
{
  PetscViewer    dataviewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename,&dataviewer); CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(dataviewer, PETSC_VIEWER_ASCII_SYMMODU); CHKERRQ(ierr);
  ierr = VecView(x,dataviewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&dataviewer); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SolutionSaveBinary"
PetscErrorCode SolutionSaveBinary(Vec x, const char filename[])
{
  PetscViewer    dataviewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerCreate(PETSC_COMM_WORLD, &dataviewer); CHKERRQ(ierr);
  ierr = PetscViewerSetType(dataviewer, PETSCVIEWERBINARY); CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(dataviewer, FILE_MODE_WRITE); CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(dataviewer, filename); CHKERRQ(ierr);
  ierr = VecView(x,dataviewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&dataviewer); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
