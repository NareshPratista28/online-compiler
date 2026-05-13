package resh8_gmail_com;

import static org.junit.Assert.*;
import org.junit.Test;

public class JUnitPeriksaBilanganTest {
    
    @Test
    public void testMainMethodExecution() {
        try {
            PeriksaBilangan.main(new String[]{});
            assertTrue("Main method dapat dijalankan tanpa error", true);
        } catch (Exception e) {
            fail("Main method error: " + e.getMessage());
        }
    }
    
    @Test
    public void testValidasinilaiCase1() {
        // Test case: Test validasiNilai case 1
        assertTrue(PeriksaBilangan.validasiNilai(85));
    }
    
    @Test
    public void testValidasinilaiCase2() {
        // Test case: Test validasiNilai case 2
        assertFalse(PeriksaBilangan.validasiNilai(-5));
    }
    
    @Test
    public void testValidasinilaiCase3() {
        // Test case: Test validasiNilai case 3
        assertFalse(PeriksaBilangan.validasiNilai(105));
    }
}