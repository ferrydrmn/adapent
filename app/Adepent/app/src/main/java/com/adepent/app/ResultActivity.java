package com.adepent.app;

import androidx.annotation.NonNull;
import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.widget.ImageView;
import android.widget.TextView;

public class ResultActivity extends AppCompatActivity {

    private TextView tv_result;
    private TextView tv_description;
    private ImageView iv_result;
    private Uri imgUri;
    private ActionBar actionBar;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        getSupportActionBar().setDisplayHomeAsUpEnabled(true);
        getSupportActionBar().setSubtitle("Aplikasi Deteksi Penyakit Daun Teh");

        tv_result = findViewById(R.id.tv_result);
        tv_description = findViewById(R.id.tv_desc);
        iv_result = findViewById(R.id.iv_result);

        String result = getIntent().getStringExtra("result");
        String description = getIntent().getStringExtra("description");
        String imgUriString = getIntent().getStringExtra("imgUri");

        tv_result.setText(result);
        tv_description.setText(description);

        imgUri = Uri.parse(imgUriString);
        iv_result.setImageURI(imgUri);
        iv_result.getLayoutParams().height = 500;
        iv_result.getLayoutParams().width = 500;

    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu, menu);

        return true;
    }

    @Override
    public boolean onSupportNavigateUp() {
        onBackPressed();
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        switch (item.getItemId()) {
            case R.id.menu_about:
                Intent intent = new Intent(ResultActivity.this, AboutActivity.class);
                startActivity(intent);
                return true;
        }
        return super.onOptionsItemSelected(item);
    }
}