import cv2
import numpy as np

def ordenar_puntos(pts):
    """
    Ordena 4 puntos geométricos en el orden estricto de homografía:
    [Arriba-Izquierda, Arriba-Derecha, Abajo-Derecha, Abajo-Izquierda]
    """
    pts = pts.reshape((4, 2))
    nuevos_pts = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    nuevos_pts[0] = pts[np.argmin(s)]
    nuevos_pts[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1).flatten()
    nuevos_pts[1] = pts[np.argmin(diff)]
    nuevos_pts[3] = pts[np.argmax(diff)]

    return nuevos_pts

def calcular_vertices_cancha_robust(mask_cancha):
    """
    Calcula de manera infalible las 4 esquinas extremas de la masa verde de la cancha
    usando Envolvente Convexa (Convex Hull).
    """
    contornos, _ = cv2.findContours(mask_cancha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contornos) == 0:
        return None
    
    cancha_principal = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(cancha_principal) < 5000:
        return None
        
    pts = cancha_principal.reshape(-1, 2)
    hull = cv2.convexHull(pts)
    pts_hull = hull.reshape(-1, 2)
    
    # Extraer extremos usando suma y diferencia
    suma = pts_hull.sum(axis=1)
    pt_arriba_izq = pts_hull[np.argmin(suma)]
    pt_abajo_der = pts_hull[np.argmax(suma)]
    
    diff = np.diff(pts_hull, axis=1).flatten()
    pt_arriba_der = pts_hull[np.argmin(diff)]
    pt_abajo_izq = pts_hull[np.argmax(diff)]
    
    return np.array([pt_arriba_izq, pt_arriba_der, pt_abajo_der, pt_abajo_izq], dtype=np.float32)

def similitud_robots(desc1, desc2):
    """Calcula la similitud coseno convertida a rango de porcentaje (0-100)"""
    desc1 = np.asarray(desc1, dtype=np.float32)
    desc2 = np.asarray(desc2, dtype=np.float32)
    norma1 = np.linalg.norm(desc1)
    norma2 = np.linalg.norm(desc2)
    if norma1 == 0 or norma2 == 0:
        return 0.0
    cos_sim = np.dot(desc1, desc2) / (norma1 * norma2)
    return (cos_sim + 1.0) * 50.0

def agrupar_robots_por_similitud(matriz_similitud):
    n = matriz_similitud.shape[0]
    visitado = np.zeros(n, dtype=bool)
    equipos = []
    for i in range(n):
        if visitado[i]:
            continue
        equipo = [i]
        visitado[i] = True
        for j in range(n):
            if i != j and not visitado[j]:
                if matriz_similitud[i, j] == np.max(matriz_similitud[i]):
                    equipo.append(j)
                    visitado[j] = True
        equipos.append(equipo)
    return equipos