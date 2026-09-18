"""
Sistema de Recomendacion de Peliculas basado en Algebra Lineal
----------------------------------------------------------------
Proyecto Final - Algebra Lineal - UMG
Unidades cubiertas: 1 (Sistemas de ecuaciones, matrices, inversa,
transpuesta) y 2 (Determinantes, adjunta, Cramer).

IDEA CENTRAL
Cada usuario califica algunas peliculas que ya vio. Cada pelicula
tiene un "vector de generos" (que tanto pertenece a Accion, Comedia
y Terror, en escala 0 a 1). Se busca un vector de pesos w (uno por
genero) que explique las calificaciones ya dadas por el usuario:

        A * w = b

donde A (m x n) trae, en cada fila, el vector de generos de una
pelicula ya vista, y b (m x 1) trae la calificacion que el usuario
le dio. Al resolver el sistema se obtiene w = "cuanto le gusta a
ese usuario cada genero". Con w ya calculado, se predice la
calificacion de cualquier pelicula nueva con un producto matricial:

        prediccion = M_catalogo @ w

y se recomiendan las peliculas con mayor prediccion.

Segun cuantas peliculas (m) haya calificado el usuario frente al
numero de generos (n = 3):
    - Si m == n y el sistema es cuadrado -> se resuelve DIRECTAMENTE
      con eliminacion Gauss-Jordan, con la inversa (por adjunta) o
      con la regla de Cramer.
    - Si m > n (mas peliculas calificadas que generos, lo mas comun)
      no hay solucion exacta en general, asi que se usa la ECUACION
      NORMAL: multiplicar ambos lados por la TRANSPUESTA de A:

            (Aᵀ A) w = Aᵀ b

      Esto reduce el sistema rectangular a un sistema CUADRADO n x n,
      que ya se puede resolver con los mismos 3 metodos anteriores.
    - Si un usuario nuevo no ha calificado nada, b = vector cero y el
      sistema A w = 0 es HOMOGENEO: si det(A) != 0 la unica solucion
      es la trivial (w = 0), es decir, todavia no hay informacion
      suficiente para personalizar sus recomendaciones.
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


# ==========================================================
# 1. Operaciones de algebra lineal implementadas manualmente
# ==========================================================
def transponer(A):
    """Transpuesta de una matriz (unidad 1.6): intercambia filas por columnas."""
    A = np.array(A, dtype=float)
    filas, columnas = A.shape
    At = np.zeros((columnas, filas))
    for i in range(filas):
        for j in range(columnas):
            At[j][i] = A[i][j]
    return At


def determinante(A):
    """Determinante de una matriz cuadrada n x n por expansion de
    cofactores (unidad 2.1). Caso base 1x1 y 2x2, recursivo para n > 2."""
    A = np.array(A, dtype=float)
    n = A.shape[0]
    if n == 1:
        return A[0, 0]
    if n == 2:
        return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    det = 0.0
    for j in range(n):
        menor = np.delete(np.delete(A, 0, axis=0), j, axis=1)
        cofactor = ((-1) ** j) * determinante(menor)
        det += A[0, j] * cofactor
    return det


def matriz_cofactores(A):
    """Matriz de cofactores: cada elemento es (-1)^(i+j) por el
    determinante del menor que resulta de tachar fila i, columna j."""
    A = np.array(A, dtype=float)
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            menor = np.delete(np.delete(A, i, axis=0), j, axis=1)
            C[i, j] = ((-1) ** (i + j)) * determinante(menor)
    return C


def inversa_por_adjunta(A):
    """Inversa de una matriz cuadrada via la adjunta (unidad 2.3):
    adjunta(A) = transpuesta de la matriz de cofactores
    A^-1 = (1 / det(A)) * adjunta(A)   (unidad 2.4: requiere det != 0)"""
    det = determinante(A)
    if abs(det) < 1e-12:
        raise ValueError("La matriz no es invertible (det = 0).")
    adjunta = transponer(matriz_cofactores(A))
    return adjunta / det


def eliminacion_gauss_jordan(A, b):
    """Resuelve A w = b llevando la matriz aumentada [A | b] a su
    forma escalonada reducida (unidad 1.2)."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    n = A.shape[0]
    aug = np.hstack([A, b])

    for col in range(n):
        # Pivoteo parcial: evita dividir por un pivote muy pequeno o cero
        fila_pivote = np.argmax(np.abs(aug[col:, col])) + col
        if abs(aug[fila_pivote, col]) < 1e-12:
            raise ValueError("El sistema no tiene solucion unica (pivote = 0).")
        aug[[col, fila_pivote]] = aug[[fila_pivote, col]]
        aug[col] = aug[col] / aug[col, col]
        for f in range(n):
            if f != col:
                aug[f] = aug[f] - aug[f, col] * aug[col]
    return aug[:, -1]


def resolver_por_cramer(A, b):
    """Regla de Cramer (unidad 2.5): w_i = det(A_i) / det(A), donde
    A_i es A con la columna i reemplazada por b."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]
    det_A = determinante(A)
    if abs(det_A) < 1e-12:
        raise ValueError("El sistema no tiene solucion unica (det = 0).")
    w = np.zeros(n)
    for i in range(n):
        Ai = A.copy()
        Ai[:, i] = b
        w[i] = determinante(Ai) / det_A
    return w


def resolver_preferencias(A, b):
    """
    Encuentra el vector de pesos w para un usuario.
    - Si A es cuadrada (m == n): resuelve A w = b directamente.
    - Si A es rectangular (m > n): arma la ecuacion normal
      (Aᵀ A) w = Aᵀ b y resuelve ese sistema cuadrado n x n.
    Devuelve w calculado por 3 caminos (Gauss-Jordan, inversa por
    adjunta y Cramer) para comprobar que coinciden.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    m, n = A.shape

    if m == n:
        A_cuadrada, b_cuadrado = A, b
    else:
        A_cuadrada = transponer(A) @ A       # Aᵀ A  (usa la transpuesta)
        b_cuadrado = transponer(A) @ b       # Aᵀ b

    det = determinante(A_cuadrada)
    if abs(det) < 1e-9:
        raise ValueError("Sistema sin solucion unica para este usuario (det = 0).")

    w_gauss = eliminacion_gauss_jordan(A_cuadrada, b_cuadrado)
    w_inversa = inversa_por_adjunta(A_cuadrada) @ b_cuadrado
    w_cramer = resolver_por_cramer(A_cuadrada, b_cuadrado)

    return {
        "w_gauss_jordan": w_gauss,
        "w_inversa_adjunta": w_inversa,
        "w_cramer": w_cramer,
        "det_sistema_cuadrado": det,
        "es_cuadrado_original": m == n,
    }


# ==========================================================
# 2. Datos: catalogo y usuarios
# ==========================================================
generos = ["Accion", "Comedia", "Terror"]

catalogo = {
    "Explosion Total":     [1.00, 0.10, 0.00],
    "Risas sin Fin":       [0.00, 1.00, 0.00],
    "Noche de Miedo":      [0.10, 0.00, 1.00],
    "Furia en la Ciudad":  [0.90, 0.00, 0.20],
    "Comedia de Enredos":  [0.10, 0.90, 0.00],
    "La Posesion":         [0.00, 0.00, 0.90],
    "Escuadron Letal":     [0.80, 0.00, 0.30],
    "Casa Embrujada":      [0.20, 0.00, 0.80],
    "Risas en el Bosque":  [0.00, 0.70, 0.10],
    "Persecucion Extrema": [0.85, 0.05, 0.05],
}

# Jose ya califico 5 peliculas (m=5 > n=3 generos) -> sistema rectangular
jose_vistas = ["Explosion Total", "Furia en la Ciudad", "Escuadron Letal",
               "Risas sin Fin", "Noche de Miedo"]
jose_ratings = [9, 8, 9, 4, 3]

# Carlos ya califico exactamente 3 peliculas (m=n=3) -> sistema cuadrado
carlos_vistas = ["Noche de Miedo", "Casa Embrujada", "Comedia de Enredos"]
carlos_ratings = [9, 8, 3]


def recomendar(w, ya_vistas, top_n=4):
    candidatos = [t for t in catalogo if t not in ya_vistas]
    predicciones = [(t, float(np.array(catalogo[t]) @ w)) for t in candidatos]
    predicciones.sort(key=lambda x: -x[1])
    return predicciones[:top_n]


# ==========================================================
# 3. Ejecucion
# ==========================================================
if __name__ == "__main__":
    for nombre, vistas, ratings in [("Jose", jose_vistas, jose_ratings),
                                     ("Carlos", carlos_vistas, carlos_ratings)]:
        A = np.array([catalogo[t] for t in vistas])
        b = np.array(ratings, dtype=float)

        print(f"===== {nombre} =====")
        print(f"Peliculas calificadas (m={A.shape[0]}) vs. generos (n={A.shape[1]})"
              f" -> sistema {'cuadrado' if A.shape[0] == A.shape[1] else 'rectangular'}")

        resultado = resolver_preferencias(A, b)
        print("det del sistema cuadrado resuelto:", round(resultado["det_sistema_cuadrado"], 4))
        print("w (Gauss-Jordan)   :", resultado["w_gauss_jordan"])
        print("w (Inversa/Adjunta):", resultado["w_inversa_adjunta"])
        print("w (Cramer)         :", resultado["w_cramer"])

        w = resultado["w_gauss_jordan"]
        print(f"\nInterpretacion de pesos -> Accion: {w[0]:.2f} | Comedia: {w[1]:.2f} | Terror: {w[2]:.2f}")

        print("\nRecomendaciones (Top 4):")
        for titulo, score in recomendar(w, vistas):
            print(f"   - {titulo:22s} prediccion: {score:.2f}")
        print()

    # ---- Caso de sistema homogeneo: usuario nuevo sin calificaciones ----
    print("===== Usuario nuevo (sin historial) =====")
    A_generico = np.array([catalogo[t] for t in carlos_vistas])
    b_cero = np.zeros(3)
    w_homogeneo = eliminacion_gauss_jordan(A_generico, b_cero)
    print("Sistema A w = 0 (homogeneo). det(A) =", round(determinante(A_generico), 4))
    print("Solucion:", w_homogeneo, "-> solucion trivial (w = 0): aun no hay",
          "informacion para personalizar; se recomienda contenido general.")
