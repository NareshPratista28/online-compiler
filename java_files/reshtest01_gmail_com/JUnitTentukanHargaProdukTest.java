package reshtest01_gmail_com;

import static org.junit.Assert.*;
import org.junit.*;
import java.io.*;

public class JUnitTentukanHargaProdukTest {
    private ByteArrayOutputStream outContent;
    private PrintStream originalOut;

    @Before
    public void setUpStreams() {
        outContent = new ByteArrayOutputStream();
        originalOut = System.out;
        System.setOut(new PrintStream(outContent));
    }

    @After
    public void restoreStreams() {
        System.setOut(originalOut);
    }

    @Test
    public void testMainMethodExecution() {
        TentukanHargaProduk.main(null);
        String output = outContent.toString().trim();
        assertFalse("Program harus menghasilkan output", output.isEmpty());
    }

    @Test
    public void testHargaElektronikTersediaBanyak() {
        TentukanHargaProduk.main(new String[]{});
        String output = outContent.toString().trim();
        assertEquals("Harga elektronik: Rp 100.000", output);
    }

    @Test
    public void testHargaFashionStokSedikit() {
        System.setProperty("kategori", "fashion");
        System.setProperty("jumlahStok", "10");
        TentukanHargaProduk.main(new String[]{});
        String output = outContent.toString().trim();
        assertEquals("Harga fashion: Rp 200.000", output);
    }

    @Test
    public void testHargaLainnyaStandar() {
        System.setProperty("kategori", "lain");
        TentukanHargaProduk.main(new String[]{});
        String output = outContent.toString().trim();
        assertEquals("Harga untuk kategori lainnya: Rp 120.000", output);
    }
}