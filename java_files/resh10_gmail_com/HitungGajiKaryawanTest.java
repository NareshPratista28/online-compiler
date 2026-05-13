package resh10_gmail_com;

import static org.junit.Assert.*;
import org.junit.Test;

public class JUnitHitungGajiKaryawanTest {

    @Test
    public void testGajiTanpaBonus() {
        // 100 * 50000 = 5.000.000 (tanpa bonus)
        double result = HitungGajiKaryawan.hitungTotalGaji(100, 50000.0);
        assertEquals(5000000.0, result, 0.001);
    }

    @Test
    public void testGajiDenganBonus() {
        // 220 * 50000 = 11.000.000 → +500.000 = 11.500.000
        double result = HitungGajiKaryawan.hitungTotalGaji(220, 50000.0);
        assertEquals(11500000.0, result, 0.001);
    }

    @Test
    public void testGajiPas10Juta() {
        // 200 * 50000 = 10.000.000 → tidak dapat bonus
        double result = HitungGajiKaryawan.hitungTotalGaji(200, 50000.0);
        assertEquals(10000000.0, result, 0.001);
    }

    @Test
    public void testGajiSedikit() {
        // 10 * 50000 = 500.000 (jauh di bawah)
        double result = HitungGajiKaryawan.hitungTotalGaji(10, 50000.0);
        assertEquals(500000.0, result, 0.001);
    }
}